import re
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import pandas as pd
from prophet import Prophet
from src.config.logging_config import setup_logging
logger = setup_logging()

class ServerMetricsPredictor:
    """
    A class to train Prophet models and predict server metrics for future dates.
    Handles multiple metrics: cpu_util, amb_temp, peak, cpu_watts, average, minimum
    Enhanced with overfitting prevention and data validation.
    """
    
    def __init__(self, server_data_raw: List[Dict]):
        self.server_data_raw = server_data_raw
        self.trained_models = {}  # Cache for trained models
        self.available_metrics = ['cpu_util', 'amb_temp', 'peak', 'cpu_watts', 'average', 'minimum']
        
        # Define realistic bounds for each metric
        # Note: cpu_util can exceed 100% for multi-core systems (e.g., 800% = 8 cores at 100%)
        self.metric_bounds = {
            'cpu_util': {'min': 0, 'max': 1600, 'unit': '%', 'description': 'Multi-core CPU utilization'},
            'amb_temp': {'min': -40, 'max': 80, 'unit': '°C', 'description': 'Ambient temperature'},
            'peak': {'min': 0, 'max': 2000, 'unit': 'W', 'description': 'Peak power consumption'},
            'cpu_watts': {'min': 0, 'max': 500, 'unit': 'W', 'description': 'CPU power consumption'},
            'average': {'min': 0, 'max': 1500, 'unit': 'W', 'description': 'Average power consumption'},
            'minimum': {'min': 0, 'max': 1000, 'unit': 'W', 'description': 'Minimum power consumption'}
        }
        
    def _parse_datetime(self, time_str: str) -> Optional[datetime]:
        """Parse various datetime formats from the server data."""
        formats = [
            "%d/%m/%Y, %H:%M:%S",
            "%Y-%m-%d %H:%M:%S",
            "%d-%m-%Y %H:%M:%S",
            "%Y/%m/%d %H:%M:%S"
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(time_str, fmt)
            except ValueError:
                continue
        
        logger.warning(f"Could not parse datetime: {time_str}")
        return None
    
    def _validate_and_clean_data(self, df: pd.DataFrame, metric: str) -> pd.DataFrame:
        """
        Validate and clean the data to prevent overfitting.
        
        Args:
            df: DataFrame with 'ds' and 'y' columns
            metric: Metric name for bounds checking
            
        Returns:
            Cleaned DataFrame
        """
        original_count = len(df)
        
        # Remove NaN values
        df = df.dropna(subset=['y'])
        
        # Get metric bounds
        bounds = self.metric_bounds.get(metric, {'min': float('-inf'), 'max': float('inf')})
        
        # Remove outliers using IQR method
        Q1 = df['y'].quantile(0.25)
        Q3 = df['y'].quantile(0.75)
        IQR = Q3 - Q1
        
        # Define outlier bounds 
        if metric == 'cpu_util':
            # For multi-core CPU utilization, use more lenient bounds
            # since high values (e.g., 800% on 8-core system) are normal
            lower_bound = max(bounds['min'], Q1 - 2.0 * IQR)
            upper_bound = min(bounds['max'], Q3 + 2.5 * IQR)
        else:
            lower_bound = max(bounds['min'], Q1 - 2.0 * IQR)
            upper_bound = min(bounds['max'], Q3 + 2.0 * IQR)
        
        # Remove outliers
        df_clean = df[(df['y'] >= lower_bound) & (df['y'] <= upper_bound)].copy()
        
        # If we removed too many points, use a more lenient approach
        if len(df_clean) < 0.7 * original_count:
            logger.warning(f"Outlier removal too aggressive for {metric}, using bounds-only filtering")
            df_clean = df[(df['y'] >= bounds['min']) & (df['y'] <= bounds['max'])].copy()
        
        # Log data cleaning results
        removed_count = original_count - len(df_clean)
        if removed_count > 0:
            logger.info(f"Cleaned {metric} data: removed {removed_count} outliers/invalid points")
            logger.info(f"Value range after cleaning: {df_clean['y'].min():.2f} - {df_clean['y'].max():.2f}")
        
        return df_clean.reset_index(drop=True)
    
    def _prepare_data_for_prophet(self, serial_number: str, metric: str) -> Optional[pd.DataFrame]:
        """
        Prepare time series data for Prophet model training.
        
        Args:
            serial_number: Server serial number
            metric: Metric to predict (cpu_util, amb_temp, peak, etc.)
            
        Returns:
            DataFrame with 'ds' (datetime) and 'y' (metric values) columns
        """
        # Find server data
        server_data = None
        for server in self.server_data_raw:
            if server.get('serial_number') == serial_number:
                server_data = server
                break
        
        if not server_data:
            logger.error(f"Server {serial_number} not found")
            return None
        
        if 'power' not in server_data or not server_data['power']:
            logger.error(f"No power data found for server {serial_number}")
            return None
        
        # Extract time series data
        data_points = []
        for record in server_data['power']:
            if 'time' not in record:
                continue
                
            timestamp = self._parse_datetime(record['time'])
            if timestamp is None:
                continue
            
            if metric in record and record[metric] is not None:
                try:
                    value = float(record[metric])
                    data_points.append({
                        'ds': timestamp,
                        'y': value
                    })
                except (ValueError, TypeError):
                    continue
        
        if len(data_points) < 20:  # Increased minimum for better model stability
            logger.error(f"Insufficient data points ({len(data_points)}) for server {serial_number}, metric {metric}")
            return None
        
        df = pd.DataFrame(data_points)
        df = df.sort_values('ds').reset_index(drop=True)
        
        # Remove duplicates, keeping the last value for each timestamp
        df = df.drop_duplicates(subset=['ds'], keep='last')
        
        # Validate and clean the data
        df = self._validate_and_clean_data(df, metric)
        
        if len(df) < 15:  # Check again after cleaning
            logger.error(f"Insufficient clean data points ({len(df)}) for server {serial_number}, metric {metric}")
            return None
        
        logger.info(f"Prepared {len(df)} clean data points for {serial_number} - {metric}")
        return df
    
    def _get_model_parameters(self, metric: str, df: pd.DataFrame) -> Dict:
        """
        Get optimized model parameters based on metric type and data characteristics.
        
        Args:
            metric: Metric name
            df: Training data
            
        Returns:
            Dictionary of Prophet model parameters
        """
        # Base parameters with conservative settings to prevent overfitting
        params = {
            'daily_seasonality': False,
            'weekly_seasonality': False,
            'yearly_seasonality': False,
            'seasonality_mode': 'additive',
            'changepoint_prior_scale': 0.05,  # More conservative (default: 0.05)
            'seasonality_prior_scale': 1.0,   # More conservative (default: 10.0)
            'holidays_prior_scale': 1.0,      # More conservative (default: 10.0)
            'mcmc_samples': 0,                # Disable MCMC for faster training
            'interval_width': 0.8,            # Narrower confidence intervals
            'uncertainty_samples': 100        # Reduce uncertainty samples
        }
        
        # Data span analysis
        time_span = (df['ds'].max() - df['ds'].min()).days
        data_frequency = len(df) / max(time_span, 1)  # points per day
        
        # Adjust seasonality based on data characteristics
        if time_span >= 7 and data_frequency >= 1:  # At least a week of data with daily frequency
            params['daily_seasonality'] = True
            
        if time_span >= 14 and data_frequency >= 0.5:  # At least 2 weeks
            params['weekly_seasonality'] = True
            
        # Metric-specific adjustments
        if metric == 'cpu_util':
            # Multi-core CPU utilization often has strong daily patterns
            params['seasonality_mode'] = 'additive'  # Better for multi-core metrics
            params['changepoint_prior_scale'] = 0.03  # Less conservative for multi-core CPU
            
        elif metric == 'amb_temp':
            # Temperature has more predictable patterns
            params['daily_seasonality'] = True
            if time_span >= 365:
                params['yearly_seasonality'] = True
                
        elif metric in ['peak', 'cpu_watts', 'average', 'minimum']:
            # Power metrics can be volatile
            params['changepoint_prior_scale'] = 0.02
            params['seasonality_prior_scale'] = 0.5
        
        return params
    
    def _train_prophet_model(self, df: pd.DataFrame, metric: str) -> Optional[Prophet]:
        """
        Train a Prophet model on the prepared data with overfitting prevention.
        
        Args:
            df: DataFrame with 'ds' and 'y' columns
            metric: Metric name for logging
            
        Returns:
            Trained Prophet model or None if training fails
        """
        try:
            # Get optimized parameters
            model_params = self._get_model_parameters(metric, df)
            
            model = Prophet(**model_params)
            
            # Add conservative custom seasonalities only if we have enough data
            if len(df) >= 48:  # At least 2 days of hourly data
                model.add_seasonality(
                    name='hourly', 
                    period=1, 
                    fourier_order=2,  # Reduced from 3 to prevent overfitting
                    prior_scale=0.1   # Very conservative
                )
            
            # Fit the model
            model.fit(df)
            logger.info(f"Successfully trained Prophet model for metric: {metric}")
            return model
            
        except Exception as e:
            logger.error(f"Failed to train Prophet model for metric {metric}: {str(e)}")
            return None
    
    def _validate_prediction(self, predicted_value: float, metric: str, 
                           historical_data: pd.DataFrame) -> Tuple[float, bool]:
        """
        Validate and potentially adjust predictions to prevent unrealistic values.
        
        Args:
            predicted_value: Raw predicted value from model
            metric: Metric name
            historical_data: Historical data for context
            
        Returns:
            Tuple of (adjusted_value, is_adjusted)
        """
        bounds = self.metric_bounds.get(metric, {'min': float('-inf'), 'max': float('inf')})
        is_adjusted = False
        
        # Hard bounds check
        if predicted_value < bounds['min']:
            predicted_value = bounds['min']
            is_adjusted = True
        elif predicted_value > bounds['max']:
            predicted_value = bounds['max']
            is_adjusted = True
        
        # Additional validation based on historical context
        hist_mean = historical_data['y'].mean()
        hist_std = historical_data['y'].std()
        hist_max = historical_data['y'].max()
        hist_min = historical_data['y'].min()
        
        # Check if prediction is extremely far from historical patterns
        # More lenient for multi-core CPU utilization
        if metric == 'cpu_util':
            std_multiplier = 6  # More lenient for multi-core CPU
        else:
            std_multiplier = 4
            
        if abs(predicted_value - hist_mean) > std_multiplier * hist_std:
            # Cap the prediction to historical range + reasonable buffer
            if predicted_value > hist_mean:
                predicted_value = min(predicted_value, hist_max * 1.5)  # Allow 50% above historical max
            else:
                predicted_value = max(predicted_value, hist_min * 0.5)
            is_adjusted = True
        
        return predicted_value, is_adjusted
    
    def predict_server_metrics(self, query: str) -> str:
        """
        Main function to predict server metrics based on natural language query.
        Enhanced with overfitting prevention and validation.
        
        Args:
            query: Natural language query like "CPU utilization for server 2M270600W3 on March 27, 2028"
            
        Returns:
            Formatted prediction results or error message
        """
        try:
            # Extract server serial number
            server_serial = self._extract_server_serial(query)
            if not server_serial:
                return self._get_available_servers_message()
            
            # Extract target date
            target_date = self._extract_target_date(query)
            if not target_date:
                return "Could not parse the target date. Please specify a date like 'March 27, 2028' or '2028-03-27'."
            
            # Extract requested metrics
            requested_metrics = self._extract_metrics_from_query(query)
            if not requested_metrics:
                requested_metrics = ['cpu_util']  # Default to CPU utilization
            
            # Generate predictions
            predictions = {}
            model_key = f"{server_serial}"
            
            for metric in requested_metrics:
                # Check if model is cached
                cache_key = f"{model_key}_{metric}"
                
                if cache_key not in self.trained_models:
                    # Prepare data and train model
                    df = self._prepare_data_for_prophet(server_serial, metric)
                    if df is None:
                        predictions[metric] = {"error": f"No sufficient clean data available for {metric}"}
                        continue
                    
                    model = self._train_prophet_model(df, metric)
                    if model is None:
                        predictions[metric] = {"error": f"Failed to train model for {metric}"}
                        continue
                    
                    # Cache the model and training data info
                    self.trained_models[cache_key] = {
                        'model': model,
                        'last_data_point': df['ds'].max(),
                        'data_points': len(df),
                        'historical_data': df  # Store for validation
                    }
                
                # Make prediction
                model_info = self.trained_models[cache_key]
                model = model_info['model']
                historical_data = model_info['historical_data']
                
                # Create future dataframe
                future_df = pd.DataFrame({'ds': [target_date]})
                forecast = model.predict(future_df)
                
                raw_predicted_value = forecast['yhat'].iloc[0]
                lower_bound = forecast['yhat_lower'].iloc[0]
                upper_bound = forecast['yhat_upper'].iloc[0]
                
                # Validate and adjust prediction
                validated_value, was_adjusted = self._validate_prediction(
                    raw_predicted_value, metric, historical_data
                )
                
                # Also validate bounds
                bounds = self.metric_bounds.get(metric, {'min': float('-inf'), 'max': float('inf')})
                lower_bound = max(lower_bound, bounds['min'])
                upper_bound = min(upper_bound, bounds['max'])
                
                predictions[metric] = {
                    'predicted_value': round(validated_value, 2),
                    'lower_bound': round(lower_bound, 2),
                    'upper_bound': round(upper_bound, 2),
                    'confidence_interval': f"[{round(lower_bound, 2)}, {round(upper_bound, 2)}]",
                    'training_data_points': model_info['data_points'],
                    'last_data_date': model_info['last_data_point'].strftime('%Y-%m-%d'),
                    'was_adjusted': was_adjusted,
                    'raw_prediction': round(raw_predicted_value, 2) if was_adjusted else None
                }
            
            return self._format_prediction_output(server_serial, target_date, predictions)
            
        except Exception as e:
            logger.error(f"Error in predict_server_metrics: {str(e)}")
            return f"An error occurred while generating predictions: {str(e)}"
    
    def _extract_server_serial(self, query: str) -> Optional[str]:
        """Extract server serial number from query."""
        # Try various patterns
        patterns = [
            r'server\s+([A-Za-z0-9_-]+)',
            r'serial\s+([A-Za-z0-9_-]+)',
            r'([A-Za-z0-9]{6,})',  # Direct alphanumeric codes
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            for match in matches:
                # Check if this serial exists in our data
                for server in self.server_data_raw:
                    if server.get('serial_number') == match:
                        return match
        return None
    
    def _extract_target_date(self, query: str) -> Optional[datetime]:
        """Extract target date from query."""
        # Try various date patterns
        date_patterns = [
            r'(\d{1,2})\s+(\w+)\s+(\d{4})',  # "27 March 2028"
            r'(\w+)\s+(\d{1,2}),?\s+(\d{4})',  # "March 27, 2028"
            r'(\d{4})-(\d{1,2})-(\d{1,2})',  # "2028-03-27"
            r'(\d{1,2})/(\d{1,2})/(\d{4})',  # "27/03/2028"
            r'(\d{1,2})-(\d{1,2})-(\d{4})',  # "27-03-2028"
        ]
        
        months = {
            'january': 1, 'february': 2, 'march': 3, 'april': 4, 'may': 5, 'june': 6,
            'july': 7, 'august': 8, 'september': 9, 'october': 10, 'november': 11, 'december': 12,
            'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
            'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
        }
        
        for pattern in date_patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            for match in matches:
                try:
                    if len(match) == 3:
                        if pattern == r'(\d{1,2})\s+(\w+)\s+(\d{4})':  # "27 March 2028"
                            day, month_str, year = match
                            month = months.get(month_str.lower())
                            if month:
                                return datetime(int(year), month, int(day))
                        elif pattern == r'(\w+)\s+(\d{1,2}),?\s+(\d{4})':  # "March 27, 2028"
                            month_str, day, year = match
                            month = months.get(month_str.lower())
                            if month:
                                return datetime(int(year), month, int(day))
                        elif pattern == r'(\d{4})-(\d{1,2})-(\d{1,2})':  # "2028-03-27"
                            year, month, day = match
                            return datetime(int(year), int(month), int(day))
                        elif pattern in [r'(\d{1,2})/(\d{1,2})/(\d{4})', r'(\d{1,2})-(\d{1,2})-(\d{4})']:
                            day, month, year = match
                            return datetime(int(year), int(month), int(day))
                except ValueError:
                    continue
        
        # Check for relative dates like "tomorrow", "next week"
        query_lower = query.lower()
        today = datetime.now()
        
        if 'tomorrow' in query_lower:
            return today + timedelta(days=1)
        elif 'next week' in query_lower:
            return today + timedelta(days=7)
        elif 'next month' in query_lower:
            return today + timedelta(days=30)
        
        return None
    
    def _extract_metrics_from_query(self, query: str) -> List[str]:
        """Extract requested metrics from query."""
        query_lower = query.lower()
        found_metrics = []
        
        metric_keywords = {
            'cpu_util': ['cpu utilization', 'cpu usage', 'cpu util', 'processor usage'],
            'amb_temp': ['ambient temperature', 'amb temp', 'temperature', 'temp'],
            'peak': ['peak power', 'peak', 'maximum power'],
            'cpu_watts': ['cpu watts', 'cpu power', 'processor power'],
            'average': ['average power', 'avg power', 'average'],
            'minimum': ['minimum power', 'min power', 'minimum']
        }
        
        for metric, keywords in metric_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                found_metrics.append(metric)
        
        return found_metrics if found_metrics else ['cpu_util']  # Default to CPU utilization
    
    def _get_available_servers_message(self) -> str:
        """Return message with available servers."""
        available_servers = [server.get('serial_number') for server in self.server_data_raw[:10]]
        return (f"Server not found. Available servers include: {', '.join(available_servers)}\n"
                f"Example usage: 'CPU utilization for server {available_servers[0]} on March 27, 2028'")
    
    def _format_prediction_output(self, server_serial: str, target_date: datetime, predictions: Dict) -> str:
        """Format the prediction results for output."""
        output = f"🔮 **Prediction Results for Server {server_serial}**\n"
        output += f"📅 **Target Date**: {target_date.strftime('%B %d, %Y')}\n\n"
        
        if not predictions:
            return output + "❌ No predictions could be generated."
        
        for metric, pred_data in predictions.items():
            if 'error' in pred_data:
                output += f"❌ **{metric.upper()}**: {pred_data['error']}\n\n"
                continue
            
            # Format metric name
            metric_names = {
                'cpu_util': 'CPU Utilization (Multi-core)',
                'amb_temp': 'Ambient Temperature',
                'peak': 'Peak Power',
                'cpu_watts': 'CPU Power',
                'average': 'Average Power',
                'minimum': 'Minimum Power'
            }
            
            bounds = self.metric_bounds.get(metric, {})
            unit = bounds.get('unit', '')
            
            metric_display = metric_names.get(metric, metric.upper())
            
            output += f"📊 **{metric_display}**:\n"
            output += f"   • Predicted Value: **{pred_data['predicted_value']}{unit}**\n"
            
            # Add interpretation for multi-core CPU utilization
            if metric == 'cpu_util' and pred_data['predicted_value'] > 100:
                cores_utilized = pred_data['predicted_value'] / 100
                output += f"   • Core Equivalent: ~{cores_utilized:.1f} cores at 100% utilization\n"
            
            output += f"   • Confidence Interval: {pred_data['confidence_interval']}{unit}\n"
            output += f"   • Training Data Points: {pred_data['training_data_points']}\n"
            output += f"   • Last Training Data: {pred_data['last_data_date']}\n"
            
            # Show adjustment info if prediction was capped
            if pred_data.get('was_adjusted', False):
                output += f"   • ⚠️ Prediction adjusted from {pred_data.get('raw_prediction', 'N/A')}{unit} to prevent unrealistic values\n"
            
            output += "\n"
        
        output += "⚠️ **Note**: Predictions are based on historical patterns and include validation to prevent unrealistic values."
        return output


# Tool function to be used with LangChain
def predict_server_metrics_tool(query: str) -> str:
    """
    Enhanced tool function for predicting server metrics using Prophet models.
    Now includes overfitting prevention and data validation.
    
    Usage examples:
    - "CPU utilization for server 2M270600W3 on March 27, 2028"
    - "Ambient temperature for server ABC123 tomorrow"
    - "Peak power and CPU utilization for server XYZ789 on 2028-12-25"
    
    Args:
        query: Natural language query containing server serial and target date
        
    Returns:
        Formatted prediction results with validation
    """
    global server_data_raw
    
    if not server_data_raw:
        return "❌ No server data available. Please ensure server data is loaded."
    
    predictor = ServerMetricsPredictor(server_data_raw)
    return predictor.predict_server_metrics(query)


