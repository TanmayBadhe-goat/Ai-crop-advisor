import React, { useState } from 'react';
import { View, StyleSheet, Image, Alert, ScrollView } from 'react-native';
import { Button, Card, Title, Paragraph, Text } from 'react-native-paper';
import * as ImagePicker from 'expo-image-picker';
import { cropService, DiseaseDetectionResponse } from '../api/cropService';

const DiseaseDetectionScreen = () => {
  const [image, setImage] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<DiseaseDetectionResponse | null>(null);

  const pickImage = async () => {
    const { status } = await ImagePicker.requestMediaLibraryPermissionsAsync();
    
    if (status !== 'granted') {
      Alert.alert('Permission Required', 'Please grant camera roll permissions to use this feature');
      return;
    }

    const result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Images,
      allowsEditing: true,
      aspect: [4, 3],
      quality: 0.8,
    });

    if (!result.canceled && result.assets && result.assets.length > 0) {
      setImage(result.assets[0].uri);
      setResult(null);
    }
  };

  const takePhoto = async () => {
    const { status } = await ImagePicker.requestCameraPermissionsAsync();
    
    if (status !== 'granted') {
      Alert.alert('Permission Required', 'Please grant camera permissions to use this feature');
      return;
    }

    const result = await ImagePicker.launchCameraAsync({
      allowsEditing: true,
      aspect: [4, 3],
      quality: 0.8,
    });

    if (!result.canceled && result.assets && result.assets.length > 0) {
      setImage(result.assets[0].uri);
      setResult(null);
    }
  };

  const analyzeImage = async () => {
    if (!image) {
      Alert.alert('No Image', 'Please select or take a photo first');
      return;
    }

    try {
      setLoading(true);
      console.log('Starting image analysis...');
      const response = await cropService.detectDisease(image);
      console.log('Analysis complete:', response);
      setResult(response);
    } catch (error: any) {
      console.error('Disease detection error:', error);
      
      let errorMessage = 'Failed to analyze image. Please try again.';
      
      if (error.message) {
        if (error.message.includes('Network error')) {
          errorMessage = 'Network error: Please check your internet connection and try again.';
        } else if (error.message.includes('Server error')) {
          errorMessage = 'Server error: The service is temporarily unavailable. Please try again later.';
        } else if (error.message.includes('timeout')) {
          errorMessage = 'Request timeout: The analysis is taking too long. Please try again.';
        } else {
          errorMessage = error.message;
        }
      }
      
      Alert.alert('Analysis Failed', errorMessage);
    } finally {
      setLoading(false);
    }
  };

  return (
    <ScrollView style={styles.container}>
      <View style={styles.content}>
        <Title style={styles.title}>Plant Disease Detection</Title>
        <Paragraph style={styles.subtitle}>
          Take a photo or upload an image of your plant to identify diseases
        </Paragraph>

        <Card style={styles.card}>
          <Card.Content>
            <View style={styles.buttonContainer}>
              <Button 
                mode="contained" 
                onPress={takePhoto} 
                style={[styles.button, styles.cameraButton]}
                icon="camera"
              >
                Take Photo
              </Button>
              <Button 
                mode="contained" 
                onPress={pickImage} 
                style={[styles.button, styles.galleryButton]}
                icon="image"
              >
                Gallery
              </Button>
            </View>

            {image && (
              <View style={styles.imageContainer}>
                <Image source={{ uri: image }} style={styles.image} />
                <Button
                  mode="contained"
                  onPress={analyzeImage}
                  loading={loading}
                  disabled={loading}
                  style={styles.analyzeButton}
                  icon="magnify"
                >
                  Analyze Plant
                </Button>
              </View>
            )}
            
            {!image && (
              <View style={styles.placeholderContainer}>
                <Text style={styles.placeholderEmoji}>📷</Text>
                <Text style={styles.placeholderText}>Take or upload a photo to analyze</Text>
              </View>
            )}
          </Card.Content>
        </Card>

        {result && (
          <Card style={styles.resultCard}>
            <Card.Content>
              <View style={styles.resultHeader}>
                <Text style={styles.emoji}>{result.disease.emoji}</Text>
                <View>
                  <Title style={styles.diseaseName}>{result.disease.name}</Title>
                  <View style={styles.severityContainer}>
                    <Text style={styles.severityLabel}>Severity:</Text>
                    <Text style={[
                      styles.severityValue,
                      result.disease.severity === 'High' ? styles.highSeverity :
                      result.disease.severity === 'Medium' ? styles.mediumSeverity :
                      styles.lowSeverity
                    ]}>
                      {result.disease.severity}
                    </Text>
                  </View>
                  <Paragraph style={styles.confidence}>
                    Confidence: {(result.disease.confidence * 100).toFixed(1)}%
                  </Paragraph>
                </View>
              </View>

              <View style={styles.diagnosisContainer}>
                <Title style={styles.diagnosisTitle}>Diagnosis</Title>
                
                <View style={styles.infoItem}>
                  <Text style={styles.infoLabel}>Description:</Text>
                  <Text style={styles.infoValue}>{result.diagnosis.description}</Text>
                </View>
                
                <View style={styles.infoItem}>
                  <Text style={styles.infoLabel}>Treatment:</Text>
                  <Text style={styles.infoValue}>{result.diagnosis.treatment}</Text>
                </View>
                
                <View style={styles.infoItem}>
                  <Text style={styles.infoLabel}>Prevention:</Text>
                  <Text style={styles.infoValue}>{result.diagnosis.prevention}</Text>
                </View>
              </View>
            </Card.Content>
          </Card>
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
  card: {
    marginBottom: 20,
    elevation: 4,
    borderRadius: 12,
  },
  buttonContainer: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 16,
  },
  button: {
    flex: 1,
    marginHorizontal: 4,
  },
  cameraButton: {
    backgroundColor: '#4CAF50',
  },
  galleryButton: {
    backgroundColor: '#8BC34A',
  },
  imageContainer: {
    alignItems: 'center',
  },
  image: {
    width: '100%',
    height: 250,
    borderRadius: 8,
    marginBottom: 16,
  },
  placeholderContainer: {
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#E8F5E9',
    height: 200,
    borderRadius: 8,
    marginVertical: 16,
  },
  placeholderEmoji: {
    fontSize: 50,
    marginBottom: 16,
  },
  placeholderText: {
    fontSize: 16,
    color: '#666',
  },
  analyzeButton: {
    backgroundColor: '#4CAF50',
    width: '100%',
  },
  resultCard: {
    marginBottom: 20,
    elevation: 4,
    borderRadius: 12,
    backgroundColor: '#E8F5E9',
  },
  resultHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 16,
  },
  emoji: {
    fontSize: 50,
    marginRight: 16,
  },
  diseaseName: {
    fontSize: 22,
    fontWeight: 'bold',
    color: '#2E7D32',
  },
  severityContainer: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  severityLabel: {
    fontSize: 14,
    marginRight: 4,
  },
  severityValue: {
    fontSize: 14,
    fontWeight: 'bold',
  },
  highSeverity: {
    color: '#D32F2F',
  },
  mediumSeverity: {
    color: '#FFA000',
  },
  lowSeverity: {
    color: '#388E3C',
  },
  confidence: {
    fontSize: 14,
    color: '#388E3C',
  },
  diagnosisContainer: {
    backgroundColor: 'white',
    borderRadius: 8,
    padding: 12,
  },
  diagnosisTitle: {
    fontSize: 18,
    marginBottom: 8,
    color: '#2E7D32',
  },
  infoItem: {
    marginBottom: 8,
    paddingBottom: 8,
    borderBottomWidth: 1,
    borderBottomColor: '#E0E0E0',
  },
  infoLabel: {
    fontWeight: 'bold',
    marginBottom: 4,
    color: '#333',
  },
  infoValue: {
    color: '#555',
  },
});

export default DiseaseDetectionScreen;