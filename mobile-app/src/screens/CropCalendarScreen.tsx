import React, { useState, useEffect } from 'react';
import { View, StyleSheet, ScrollView, TouchableOpacity, Dimensions } from 'react-native';
import { Card, Title, Paragraph, Text, Button, Chip } from 'react-native-paper';
import { MaterialCommunityIcons } from '@expo/vector-icons';

const { width } = Dimensions.get('window');

interface CropSeason {
  crop: string;
  emoji: string;
  season: string;
  plantingMonths: string[];
  harvestMonths: string[];
  duration: string;
  tips: string;
  color: string;
}

const CropCalendarScreen = ({ navigation }: any) => {
  const [selectedMonth, setSelectedMonth] = useState(new Date().getMonth());
  const [selectedSeason, setSelectedSeason] = useState<'All' | 'Kharif' | 'Rabi' | 'Zaid'>('All');

  const months = [
    'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
    'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'
  ];

  const seasons = ['All', 'Kharif', 'Rabi', 'Zaid'];

  const cropCalendar: CropSeason[] = [
    {
      crop: 'Rice',
      emoji: '🌾',
      season: 'Kharif',
      plantingMonths: ['Jun', 'Jul', 'Aug'],
      harvestMonths: ['Oct', 'Nov', 'Dec'],
      duration: '120-150 days',
      tips: 'Plant during monsoon. Requires flooded fields.',
      color: '#4CAF50'
    },
    {
      crop: 'Wheat',
      emoji: '🌾',
      season: 'Rabi',
      plantingMonths: ['Nov', 'Dec', 'Jan'],
      harvestMonths: ['Mar', 'Apr', 'May'],
      duration: '120-150 days',
      tips: 'Plant in winter. Requires cool weather for growth.',
      color: '#FF9800'
    },
    {
      crop: 'Maize',
      emoji: '🌽',
      season: 'Kharif',
      plantingMonths: ['Jun', 'Jul'],
      harvestMonths: ['Sep', 'Oct'],
      duration: '90-120 days',
      tips: 'Can be grown year-round with irrigation.',
      color: '#FFC107'
    },
    {
      crop: 'Cotton',
      emoji: '🌿',
      season: 'Kharif',
      plantingMonths: ['Apr', 'May', 'Jun'],
      harvestMonths: ['Oct', 'Nov', 'Dec'],
      duration: '180-200 days',
      tips: 'Requires warm weather and moderate rainfall.',
      color: '#E91E63'
    },
    {
      crop: 'Sugarcane',
      emoji: '🎋',
      season: 'Year-round',
      plantingMonths: ['Feb', 'Mar', 'Oct', 'Nov'],
      harvestMonths: ['Dec', 'Jan', 'Feb', 'Mar'],
      duration: '12-18 months',
      tips: 'Long duration crop. Plant in spring or autumn.',
      color: '#9C27B0'
    },
    {
      crop: 'Potato',
      emoji: '🥔',
      season: 'Rabi',
      plantingMonths: ['Oct', 'Nov', 'Dec'],
      harvestMonths: ['Jan', 'Feb', 'Mar'],
      duration: '90-120 days',
      tips: 'Cool weather crop. Avoid frost during harvest.',
      color: '#795548'
    },
    {
      crop: 'Tomato',
      emoji: '🍅',
      season: 'Year-round',
      plantingMonths: ['Jun', 'Jul', 'Oct', 'Nov'],
      harvestMonths: ['Sep', 'Oct', 'Jan', 'Feb'],
      duration: '90-120 days',
      tips: 'Can be grown in multiple seasons with proper care.',
      color: '#F44336'
    },
    {
      crop: 'Mustard',
      emoji: '🌻',
      season: 'Rabi',
      plantingMonths: ['Oct', 'Nov'],
      harvestMonths: ['Feb', 'Mar'],
      duration: '90-110 days',
      tips: 'Cool season oilseed crop.',
      color: '#FFEB3B'
    }
  ];

  const getFilteredCrops = () => {
    let filtered = cropCalendar;
    
    if (selectedSeason !== 'All') {
      filtered = filtered.filter(crop => 
        crop.season === selectedSeason || crop.season === 'Year-round'
      );
    }
    
    return filtered;
  };

  const getCropsForMonth = (monthIndex: number) => {
    const monthName = months[monthIndex];
    return cropCalendar.filter(crop => 
      crop.plantingMonths.includes(monthName) || crop.harvestMonths.includes(monthName)
    );
  };

  const getCurrentMonthActivity = () => {
    const currentMonth = months[selectedMonth];
    const cropsToPlant = cropCalendar.filter(crop => 
      crop.plantingMonths.includes(currentMonth)
    );
    const cropsToHarvest = cropCalendar.filter(crop => 
      crop.harvestMonths.includes(currentMonth)
    );
    
    return { cropsToPlant, cropsToHarvest };
  };

  const { cropsToPlant, cropsToHarvest } = getCurrentMonthActivity();

  return (
    <ScrollView style={styles.container}>
      <View style={styles.content}>
        <Title style={styles.title}>Crop Calendar</Title>
        <Paragraph style={styles.subtitle}>
          Plan your farming activities throughout the year
        </Paragraph>

        {/* Month Selector */}
        <Card style={styles.sectionCard}>
          <Card.Content>
            <Title style={styles.sectionTitle}>Select Month</Title>
            <ScrollView horizontal showsHorizontalScrollIndicator={false}>
              <View style={styles.monthContainer}>
                {months.map((month, index) => (
                  <TouchableOpacity
                    key={month}
                    style={[
                      styles.monthChip,
                      selectedMonth === index && styles.selectedMonthChip
                    ]}
                    onPress={() => setSelectedMonth(index)}
                  >
                    <Text style={[
                      styles.monthText,
                      selectedMonth === index && styles.selectedMonthText
                    ]}>
                      {month}
                    </Text>
                  </TouchableOpacity>
                ))}
              </View>
            </ScrollView>
          </Card.Content>
        </Card>

        {/* Season Filter */}
        <Card style={styles.sectionCard}>
          <Card.Content>
            <Title style={styles.sectionTitle}>Filter by Season</Title>
            <View style={styles.seasonContainer}>
              {seasons.map((season) => (
                <Chip
                  key={season}
                  selected={selectedSeason === season}
                  onPress={() => setSelectedSeason(season as any)}
                  style={styles.seasonChip}
                  textStyle={styles.seasonText}
                >
                  {season}
                </Chip>
              ))}
            </View>
          </Card.Content>
        </Card>

        {/* Current Month Activities */}
        <Card style={styles.sectionCard}>
          <Card.Content>
            <Title style={styles.sectionTitle}>
              Activities for {months[selectedMonth]}
            </Title>
            
            {cropsToPlant.length > 0 && (
              <View style={styles.activitySection}>
                <View style={styles.activityHeader}>
                  <MaterialCommunityIcons name="seed" size={20} color="#4CAF50" />
                  <Text style={styles.activityTitle}>Time to Plant</Text>
                </View>
                <View style={styles.cropList}>
                  {cropsToPlant.map((crop, index) => (
                    <View key={index} style={styles.cropItem}>
                      <Text style={styles.cropEmoji}>{crop.emoji}</Text>
                      <View style={styles.cropInfo}>
                        <Text style={styles.cropName}>{crop.crop}</Text>
                        <Text style={styles.cropDuration}>{crop.duration}</Text>
                      </View>
                    </View>
                  ))}
                </View>
              </View>
            )}

            {cropsToHarvest.length > 0 && (
              <View style={styles.activitySection}>
                <View style={styles.activityHeader}>
                  <MaterialCommunityIcons name="harvest" size={20} color="#FF9800" />
                  <Text style={styles.activityTitle}>Time to Harvest</Text>
                </View>
                <View style={styles.cropList}>
                  {cropsToHarvest.map((crop, index) => (
                    <View key={index} style={styles.cropItem}>
                      <Text style={styles.cropEmoji}>{crop.emoji}</Text>
                      <View style={styles.cropInfo}>
                        <Text style={styles.cropName}>{crop.crop}</Text>
                        <Text style={styles.cropDuration}>{crop.duration}</Text>
                      </View>
                    </View>
                  ))}
                </View>
              </View>
            )}

            {cropsToPlant.length === 0 && cropsToHarvest.length === 0 && (
              <View style={styles.noActivity}>
                <MaterialCommunityIcons name="calendar-blank" size={48} color="#ccc" />
                <Text style={styles.noActivityText}>
                  No major planting or harvesting activities for {months[selectedMonth]}
                </Text>
              </View>
            )}
          </Card.Content>
        </Card>

        {/* All Crops */}
        <Card style={styles.sectionCard}>
          <Card.Content>
            <Title style={styles.sectionTitle}>Crop Calendar Overview</Title>
            <View style={styles.cropsGrid}>
              {getFilteredCrops().map((crop, index) => (
                <Card key={index} style={[styles.cropCard, { borderLeftColor: crop.color }]}>
                  <Card.Content style={styles.cropCardContent}>
                    <View style={styles.cropHeader}>
                      <Text style={styles.cropCardEmoji}>{crop.emoji}</Text>
                      <View style={styles.cropCardInfo}>
                        <Text style={styles.cropCardName}>{crop.crop}</Text>
                        <Text style={styles.cropSeason}>{crop.season}</Text>
                      </View>
                    </View>
                    
                    <View style={styles.cropDetails}>
                      <View style={styles.cropDetailRow}>
                        <MaterialCommunityIcons name="seed" size={16} color="#4CAF50" />
                        <Text style={styles.cropDetailText}>
                          Plant: {crop.plantingMonths.join(', ')}
                        </Text>
                      </View>
                      <View style={styles.cropDetailRow}>
                        <MaterialCommunityIcons name="harvest" size={16} color="#FF9800" />
                        <Text style={styles.cropDetailText}>
                          Harvest: {crop.harvestMonths.join(', ')}
                        </Text>
                      </View>
                      <View style={styles.cropDetailRow}>
                        <MaterialCommunityIcons name="clock" size={16} color="#666" />
                        <Text style={styles.cropDetailText}>
                          Duration: {crop.duration}
                        </Text>
                      </View>
                    </View>
                    
                    <Text style={styles.cropTips}>{crop.tips}</Text>
                  </Card.Content>
                </Card>
              ))}
            </View>
          </Card.Content>
        </Card>

        <View style={styles.bottomSpacing} />
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
    textAlign: 'center',
    marginBottom: 8,
    color: '#333',
  },
  subtitle: {
    fontSize: 16,
    textAlign: 'center',
    color: '#666',
    marginBottom: 20,
  },
  sectionCard: {
    marginBottom: 16,
    elevation: 2,
    borderRadius: 12,
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#333',
    marginBottom: 12,
  },
  monthContainer: {
    flexDirection: 'row',
    paddingVertical: 8,
  },
  monthChip: {
    paddingHorizontal: 16,
    paddingVertical: 8,
    marginRight: 8,
    backgroundColor: '#F0F0F0',
    borderRadius: 20,
  },
  selectedMonthChip: {
    backgroundColor: '#4CAF50',
  },
  monthText: {
    fontSize: 14,
    color: '#333',
    fontWeight: '500',
  },
  selectedMonthText: {
    color: 'white',
  },
  seasonContainer: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 8,
  },
  seasonChip: {
    marginRight: 8,
    marginBottom: 8,
  },
  seasonText: {
    fontSize: 12,
  },
  activitySection: {
    marginBottom: 20,
  },
  activityHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 12,
  },
  activityTitle: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#333',
    marginLeft: 8,
  },
  cropList: {
    backgroundColor: '#F8F9FA',
    borderRadius: 8,
    padding: 12,
  },
  cropItem: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: 8,
    borderBottomWidth: 1,
    borderBottomColor: '#E0E0E0',
  },
  cropEmoji: {
    fontSize: 24,
    marginRight: 12,
  },
  cropInfo: {
    flex: 1,
  },
  cropName: {
    fontSize: 16,
    fontWeight: '500',
    color: '#333',
  },
  cropDuration: {
    fontSize: 12,
    color: '#666',
    marginTop: 2,
  },
  noActivity: {
    alignItems: 'center',
    paddingVertical: 32,
  },
  noActivityText: {
    fontSize: 14,
    color: '#666',
    textAlign: 'center',
    marginTop: 12,
  },
  cropsGrid: {
    marginTop: 8,
  },
  cropCard: {
    marginBottom: 12,
    borderLeftWidth: 4,
    elevation: 1,
  },
  cropCardContent: {
    paddingVertical: 12,
  },
  cropHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 12,
  },
  cropCardEmoji: {
    fontSize: 32,
    marginRight: 12,
  },
  cropCardInfo: {
    flex: 1,
  },
  cropCardName: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#333',
  },
  cropSeason: {
    fontSize: 12,
    color: '#666',
    marginTop: 2,
  },
  cropDetails: {
    marginBottom: 12,
  },
  cropDetailRow: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 4,
  },
  cropDetailText: {
    fontSize: 12,
    color: '#555',
    marginLeft: 8,
  },
  cropTips: {
    fontSize: 12,
    color: '#666',
    fontStyle: 'italic',
    backgroundColor: '#F8F9FA',
    padding: 8,
    borderRadius: 6,
  },
  bottomSpacing: {
    height: 20,
  },
});

export default CropCalendarScreen;
