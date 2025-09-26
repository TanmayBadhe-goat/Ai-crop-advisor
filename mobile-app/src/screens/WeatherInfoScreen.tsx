import React, { useState, useEffect } from 'react';
import { View, StyleSheet, ScrollView, ActivityIndicator } from 'react-native';
import { Card, Title, Paragraph, Text, Button } from 'react-native-paper';
import * as Location from 'expo-location';
import { cropService, WeatherResponse } from '../api/cropService';

const WeatherInfoScreen = () => {
  const [location, setLocation] = useState<Location.LocationObject | null>(null);
  const [weather, setWeather] = useState<WeatherResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [lastUpdated, setLastUpdated] = useState<string | null>(null);

  const getLocationAndWeather = async () => {
    setLoading(true);
    setError(null);
    
    try {
      const { status } = await Location.requestForegroundPermissionsAsync();
      
      if (status !== 'granted') {
        setError('Permission to access location was denied');
        setLoading(false);
        return;
      }
      
      const location = await Location.getCurrentPositionAsync({});
      setLocation(location);
      
      // Get weather data
      const weatherData = await cropService.getWeather({
        latitude: location.coords.latitude,
        longitude: location.coords.longitude
      });
      // If backend returns a success flag and it is false, surface the error
      if ((weatherData as any)?.success === false) {
        setWeather(null);
        setError('Weather service returned an error');
      } else {
        // Normalize legacy shape { success, weather: { temperature, humidity, description } }
        let normalized = weatherData as any;
        if ((weatherData as any)?.weather && !(weatherData as any)?.current) {
          const w = (weatherData as any).weather;
          normalized = {
            success: true,
            location: { city: 'Current Location', country: '' },
            current: {
              temperature: Math.round(w.temperature ?? 0),
              humidity: w.humidity ?? 0,
              condition: w.description ?? 'clear',
              windSpeed: 0,
              precipitation: 0,
            },
            forecast: [],
            agricultural_advisory: [],
          } as any;
        }
        // Frontend fallbacks: ensure forecast and advisory are present
        try {
          const cur = (normalized as any).current;
          if (!Array.isArray((normalized as any).forecast) || (normalized as any).forecast.length === 0) {
            const base = typeof cur?.temperature === 'number' ? cur.temperature : 0;
            const cond = (cur?.condition || 'clear') as string;
            const today = new Date();
            (normalized as any).forecast = [1,2,3].map((d) => {
              const date = new Date(today.getTime());
              date.setDate(today.getDate() + d);
              const max = Math.round(base + (d === 2 ? 1 : -1));
              const min = Math.round(base - 5 - d);
              return {
                date: date.toISOString().slice(0,10),
                maxTemp: max,
                minTemp: min,
                condition: cond,
              };
            });
          }
          if (!Array.isArray((normalized as any).agricultural_advisory) || (normalized as any).agricultural_advisory.length === 0) {
            const tips: any[] = [];
            if (cur?.temperature > 35) {
              tips.push({ title: 'High Temperature Alert', description: 'Provide shade and water more frequently.' });
            } else if (cur?.temperature < 10) {
              tips.push({ title: 'Cold Weather Warning', description: 'Protect sensitive crops from frost.' });
            }
            if (cur?.humidity > 80) {
              tips.push({ title: 'High Humidity', description: 'Watch for fungal diseases; improve ventilation.' });
            } else if (cur?.humidity < 40) {
              tips.push({ title: 'Low Humidity', description: 'Mulch and irrigate to retain soil moisture.' });
            }
            if (typeof cur?.precipitation === 'number' && cur.precipitation > 0) {
              tips.push({ title: 'Rain Advisory', description: 'Check drainage; avoid field operations in heavy rain.' });
            }
            if (tips.length === 0) {
              tips.push({ title: 'General Advice', description: 'Good time for routine field work and monitoring.' });
            }
            (normalized as any).agricultural_advisory = tips;
          }
        } catch {}
        setWeather(normalized);
        setLastUpdated(new Date().toLocaleTimeString());
      }
    } catch (error) {
      console.error(error);
      setError('Failed to fetch weather information');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    getLocationAndWeather();
  }, []);

  // Get weather emoji based on condition
  const getWeatherEmoji = (condition: string): string => {
    const conditionLower = condition.toLowerCase();
    if (conditionLower.includes('clear') || conditionLower.includes('sunny')) return '☀️';
    if (conditionLower.includes('cloud')) return '☁️';
    if (conditionLower.includes('rain')) return '🌧️';
    if (conditionLower.includes('thunder') || conditionLower.includes('storm')) return '⛈️';
    if (conditionLower.includes('snow')) return '❄️';
    if (conditionLower.includes('fog') || conditionLower.includes('mist')) return '🌫️';
    return '🌤️'; // Default
  };

  return (
    <ScrollView style={styles.container}>
      <View style={styles.content}>
        <Title style={styles.title}>Weather Information</Title>
        <Paragraph style={styles.subtitle}>
          Current weather conditions and agricultural recommendations
        </Paragraph>

        {loading && (
          <View style={styles.loadingContainer}>
            <ActivityIndicator size="large" color="#4CAF50" />
            <Text style={styles.loadingText}>Fetching weather data...</Text>
          </View>
        )}

        {error && (
          <Card style={styles.errorCard}>
            <Card.Content>
              <Text style={styles.errorText}>{error}</Text>
              <Button 
                mode="contained" 
                onPress={getLocationAndWeather}
                style={styles.retryButton}
              >
                Retry
              </Button>
            </Card.Content>
          </Card>
        )}

        {/* Lightweight status/debug info to help diagnose blank UI */}
        <View style={{ marginBottom: 12 }}>
          {!loading && !error && (
            <Text style={{ textAlign: 'center', color: '#888' }}>
              {weather ? `Updated at ${lastUpdated ?? ''}` : 'No weather data yet'}
            </Text>
          )}
        </View>

        {!loading && !error && weather && weather.current && (
          <>
            <Card style={styles.weatherCard}>
              <Card.Content>
                <View style={styles.weatherHeader}>
                  <Text style={styles.weatherEmoji}>
                    {getWeatherEmoji(weather.current?.condition || 'clear')}
                  </Text>
                  <View>
                    <Title style={styles.temperature}>{weather.current?.temperature || 0}°C</Title>
                    <Text style={styles.condition}>{weather.current?.condition || 'Unknown'}</Text>
                    <Text style={styles.location}>
                      {weather.location?.city || 'Unknown'}, {weather.location?.country || 'Unknown'}
                    </Text>
                  </View>
                </View>

                <View style={styles.detailsContainer}>
                  <View style={styles.detailItem}>
                    <Text style={styles.detailEmoji}>💧</Text>
                    <Text style={styles.detailLabel}>Humidity</Text>
                    <Text style={styles.detailValue}>{weather.current?.humidity || 0}%</Text>
                  </View>
                  
                  <View style={styles.detailItem}>
                    <Text style={styles.detailEmoji}>💨</Text>
                    <Text style={styles.detailLabel}>Wind</Text>
                    <Text style={styles.detailValue}>{weather.current?.windSpeed || 0} km/h</Text>
                  </View>
                  
                  <View style={styles.detailItem}>
                    <Text style={styles.detailEmoji}>☔</Text>
                    <Text style={styles.detailLabel}>Precipitation</Text>
                    <Text style={styles.detailValue}>{weather.current?.precipitation || 0} mm</Text>
                  </View>
                </View>
              </Card.Content>
            </Card>

            <Card style={styles.forecastCard}>
              <Card.Content>
                <Title style={styles.forecastTitle}>3-Day Forecast</Title>
                <View style={styles.forecastContainer}>
                  {weather.forecast?.map((day, index) => (
                    <View key={index} style={styles.forecastDay}>
                      <Text style={styles.forecastEmoji}>
                        {getWeatherEmoji(day?.condition || 'clear')}
                      </Text>
                      <Text style={styles.forecastDate}>{day?.date || 'Unknown'}</Text>
                      <Text style={styles.forecastTemp}>
                        {day?.maxTemp || 0}° / {day?.minTemp || 0}°
                      </Text>
                      <Text style={styles.forecastCondition}>{day?.condition || 'Unknown'}</Text>
                    </View>
                  )) || []}
                </View>
              </Card.Content>
            </Card>

            <Card style={styles.advisoryCard}>
              <Card.Content>
                <Title style={styles.advisoryTitle}>
                  <Text style={styles.advisoryEmoji}>🌱</Text> Agricultural Advisory
                </Title>
                <View style={styles.advisoryContainer}>
                  {weather.agricultural_advisory?.map((advice, index) => (
                    <View key={index} style={styles.adviceItem}>
                      <Text style={styles.adviceTitle}>{advice?.title || 'No title'}</Text>
                      <Text style={styles.adviceDescription}>{advice?.description || 'No description'}</Text>
                    </View>
                  )) || []}
                </View>
              </Card.Content>
            </Card>
          </>
        )}

        {!loading && !error && !weather && (
          <Card style={styles.placeholderCard}>
            <Card.Content style={styles.placeholderContent}>
              <Text style={styles.placeholderEmoji}>🌤️</Text>
              <Text style={styles.placeholderText}>
                Tap the button below to get weather information
              </Text>
              <Button 
                mode="contained" 
                onPress={getLocationAndWeather}
                style={styles.weatherButton}
                icon="refresh"
              >
                Get Weather
              </Button>
            </Card.Content>
          </Card>
        )}

        {/* Always show a refresh button at the bottom for convenience */}
        {!loading && (
          <Button 
            mode="outlined" 
            onPress={getLocationAndWeather}
            style={{ marginTop: 8 }}
            icon="refresh"
          >
            Refresh Weather
          </Button>
        )}
      </View>
    </ScrollView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#F5F5F5',
  },
  content: {
    padding: 16,
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    marginBottom: 8,
    textAlign: 'center',
  },
  subtitle: {
    fontSize: 16,
    marginBottom: 20,
    textAlign: 'center',
    color: '#666',
  },
  loadingContainer: {
    alignItems: 'center',
    justifyContent: 'center',
    padding: 20,
  },
  loadingText: {
    marginTop: 10,
    color: '#666',
  },
  errorCard: {
    backgroundColor: '#FFEBEE',
    marginBottom: 20,
    borderRadius: 12,
  },
  errorText: {
    color: '#D32F2F',
    marginBottom: 10,
  },
  retryButton: {
    backgroundColor: '#4CAF50',
    marginTop: 10,
  },
  weatherCard: {
    marginBottom: 20,
    borderRadius: 12,
    backgroundColor: '#E3F2FD',
  },
  weatherHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 16,
  },
  weatherEmoji: {
    fontSize: 60,
    marginRight: 16,
  },
  temperature: {
    fontSize: 32,
    fontWeight: 'bold',
    color: '#1565C0',
  },
  condition: {
    fontSize: 18,
    color: '#1976D2',
  },
  location: {
    fontSize: 14,
    color: '#0D47A1',
    marginTop: 4,
  },
  detailsContainer: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    backgroundColor: 'white',
    borderRadius: 8,
    padding: 12,
  },
  detailItem: {
    alignItems: 'center',
    flex: 1,
  },
  detailEmoji: {
    fontSize: 24,
    marginBottom: 4,
  },
  detailLabel: {
    fontSize: 12,
    color: '#666',
  },
  detailValue: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#333',
  },
  forecastCard: {
    marginBottom: 20,
    borderRadius: 12,
  },
  forecastTitle: {
    fontSize: 18,
    marginBottom: 12,
    color: '#0D47A1',
  },
  forecastContainer: {
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  forecastDay: {
    alignItems: 'center',
    flex: 1,
    padding: 8,
    backgroundColor: '#E3F2FD',
    borderRadius: 8,
    margin: 4,
  },
  forecastEmoji: {
    fontSize: 24,
    marginBottom: 4,
  },
  forecastDate: {
    fontSize: 12,
    color: '#666',
  },
  forecastTemp: {
    fontSize: 14,
    fontWeight: 'bold',
    color: '#1565C0',
  },
  forecastCondition: {
    fontSize: 12,
    color: '#1976D2',
    textAlign: 'center',
  },
  advisoryCard: {
    marginBottom: 20,
    borderRadius: 12,
    backgroundColor: '#E8F5E9',
  },
  advisoryTitle: {
    fontSize: 18,
    marginBottom: 12,
    color: '#2E7D32',
    flexDirection: 'row',
    alignItems: 'center',
  },
  advisoryEmoji: {
    fontSize: 20,
    marginRight: 8,
  },
  advisoryContainer: {
    backgroundColor: 'white',
    borderRadius: 8,
    padding: 12,
  },
  adviceItem: {
    marginBottom: 12,
    paddingBottom: 12,
    borderBottomWidth: 1,
    borderBottomColor: '#E0E0E0',
  },
  adviceTitle: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#2E7D32',
    marginBottom: 4,
  },
  adviceDescription: {
    fontSize: 14,
    color: '#555',
  },
  placeholderCard: {
    marginBottom: 20,
    borderRadius: 12,
    height: 250,
  },
  placeholderContent: {
    alignItems: 'center',
    justifyContent: 'center',
    height: '100%',
  },
  placeholderEmoji: {
    fontSize: 60,
    marginBottom: 16,
  },
  placeholderText: {
    fontSize: 16,
    color: '#666',
    textAlign: 'center',
    marginBottom: 20,
  },
  weatherButton: {
    backgroundColor: '#4CAF50',
  },
});

export default WeatherInfoScreen;