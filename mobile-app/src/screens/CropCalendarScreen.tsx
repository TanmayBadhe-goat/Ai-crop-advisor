import React, { useState, useEffect } from 'react';
import { View, StyleSheet, ScrollView, TouchableOpacity, Dimensions } from 'react-native';
import { Card, Title, Paragraph, Text, Button, Chip, Divider } from 'react-native-paper';
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
  yield?: string;
  marketPrice?: string;
}

interface CalendarEvent {
  id: string;
  title: string;
  date: string;
  type: "sowing" | "irrigation" | "fertilizer" | "harvest";
  crop: string;
  description: string;
  priority: "High" | "Medium" | "Low";
}

const CropCalendarScreen = ({ navigation }: any) => {
  const [selectedMonth, setSelectedMonth] = useState(new Date().getMonth());
  const [selectedSeason, setSelectedSeason] = useState<'All' | 'Kharif' | 'Rabi' | 'Zaid'>('All');

  const months = [
    'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
    'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'
  ];

  const seasons = ['All', 'Kharif', 'Rabi', 'Zaid'];

  const upcomingEvents: CalendarEvent[] = [
    {
      id: "1",
      title: "Rice Transplanting",
      date: "Tomorrow",
      type: "sowing",
      crop: "Rice",
      description: "Optimal time for rice transplanting in Kharif season",
      priority: "High"
    },
    {
      id: "2",
      title: "Wheat Irrigation",
      date: "In 3 days",
      type: "irrigation",
      crop: "Wheat",
      description: "Second irrigation recommended for wheat crop",
      priority: "High"
    },
    {
      id: "3",
      title: "Cotton Fertilizer",
      date: "In 5 days",
      type: "fertilizer",
      crop: "Cotton",
      description: "Apply NPK fertilizer for cotton flowering stage",
      priority: "Medium"
    },
    {
      id: "4",
      title: "Tomato Harvest",
      date: "In 1 week",
      type: "harvest",
      crop: "Tomato",
      description: "First harvest of tomato crop ready",
      priority: "High"
    }
  ];

  const getEventIcon = (type: string) => {
    switch (type) {
      case "sowing": return "seed";
      case "irrigation": return "water";
      case "fertilizer": return "flask";
      case "harvest": return "scissors-cutting";
      default: return "calendar";
    }
  };

  const getEventColor = (type: string) => {
    switch (type) {
      case "sowing": return "#4CAF50";
      case "irrigation": return "#2196F3";
      case "fertilizer": return "#FF9800";
      case "harvest": return "#F44336";
      default: return "#666";
    }
  };

  const cropCalendar: CropSeason[] = [
    {
      crop: 'Rice',
      emoji: '🌾',
      season: 'Kharif',
      plantingMonths: ['Jun', 'Jul', 'Aug'],
      harvestMonths: ['Oct', 'Nov', 'Dec'],
      duration: '120-150 days',
      tips: 'Plant during monsoon. Requires flooded fields.',
      color: '#4CAF50',
      yield: '3-4 tonnes/hectare',
      marketPrice: '₹2000-2500/quintal'
    },
    {
      crop: 'Wheat',
      emoji: '🌾',
      season: 'Rabi',
      plantingMonths: ['Nov', 'Dec', 'Jan'],
      harvestMonths: ['Mar', 'Apr', 'May'],
      duration: '120-150 days',
      tips: 'Plant in winter. Requires cool weather for growth.',
      color: '#FF9800',
      yield: '3-4 tonnes/hectare',
      marketPrice: '₹2100-2400/quintal'
    },
    {
      crop: 'Maize',
      emoji: '🌽',
      season: 'Kharif',
      plantingMonths: ['Jun', 'Jul'],
      harvestMonths: ['Sep', 'Oct'],
      duration: '90-120 days',
      tips: 'Can be grown year-round with irrigation.',
      color: '#FFC107',
      yield: '5-8 tonnes/hectare',
      marketPrice: '₹1800-2200/quintal'
    },
    {
      crop: 'Cotton',
      emoji: '🌿',
      season: 'Kharif',
      plantingMonths: ['Apr', 'May', 'Jun'],
      harvestMonths: ['Oct', 'Nov', 'Dec'],
      duration: '180-200 days',
      tips: 'Requires warm weather and moderate rainfall.',
      color: '#E91E63',
      yield: '1.5-2 tonnes/hectare',
      marketPrice: '₹5500-6500/quintal'
    },
    {
      crop: 'Sugarcane',
      emoji: '🎋',
      season: 'Year-round',
      plantingMonths: ['Feb', 'Mar', 'Oct', 'Nov'],
      harvestMonths: ['Dec', 'Jan', 'Feb', 'Mar'],
      duration: '12-18 months',
      tips: 'Long duration crop. Plant in spring or autumn.',
      color: '#9C27B0',
      yield: '70-80 tonnes/hectare',
      marketPrice: '₹300-350/quintal'
    },
    {
      crop: 'Potato',
      emoji: '🥔',
      season: 'Rabi',
      plantingMonths: ['Oct', 'Nov', 'Dec'],
      harvestMonths: ['Jan', 'Feb', 'Mar'],
      duration: '90-120 days',
      tips: 'Cool weather crop. Avoid frost during harvest.',
      color: '#795548',
      yield: '25-40 tonnes/hectare',
      marketPrice: '₹800-1500/quintal'
    },
    {
      crop: 'Tomato',
      emoji: '🍅',
      season: 'Year-round',
      plantingMonths: ['Jun', 'Jul', 'Oct', 'Nov'],
      harvestMonths: ['Sep', 'Oct', 'Jan', 'Feb'],
      duration: '90-120 days',
      tips: 'Can be grown in multiple seasons with proper care.',
      color: '#F44336',
      yield: '40-60 tonnes/hectare',
      marketPrice: '₹1000-2000/quintal'
    },
    {
      crop: 'Mustard',
      emoji: '🌻',
      season: 'Rabi',
      plantingMonths: ['Oct', 'Nov'],
      harvestMonths: ['Feb', 'Mar'],
      duration: '90-110 days',
      tips: 'Cool season oilseed crop.',
      color: '#FFEB3B',
      yield: '1-1.5 tonnes/hectare',
      marketPrice: '₹4500-5500/quintal'
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
                  <MaterialCommunityIcons name="scissors-cutting" size={20} color="#FF9800" />
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

        {/* Upcoming Events */}
        <Card style={styles.sectionCard}>
          <Card.Content>
            <Title style={styles.sectionTitle}>Upcoming Events</Title>
            <View style={styles.eventsContainer}>
              {upcomingEvents.map((event) => (
                <View key={event.id} style={styles.eventItem}>
                  <View style={[styles.eventIcon, { backgroundColor: getEventColor(event.type) + '20' }]}>
                    <MaterialCommunityIcons 
                      name={getEventIcon(event.type) as any} 
                      size={20} 
                      color={getEventColor(event.type)} 
                    />
                  </View>
                  <View style={styles.eventContent}>
                    <View style={styles.eventHeader}>
                      <Text style={styles.eventTitle}>{event.title}</Text>
                      <View style={[styles.priorityBadge, { 
                        backgroundColor: event.priority === 'High' ? '#FF5722' : 
                                       event.priority === 'Medium' ? '#FF9800' : '#4CAF50' 
                      }]}>
                        <Text style={styles.priorityText}>{event.crop}</Text>
                      </View>
                    </View>
                    <Text style={styles.eventDescription}>{event.description}</Text>
                    <View style={styles.eventFooter}>
                      <Text style={styles.eventDate}>{event.date}</Text>
                      <Text style={styles.eventType}>{event.type.toUpperCase()}</Text>
                    </View>
                  </View>
                </View>
              ))}
            </View>
          </Card.Content>
        </Card>

        {/* Weather-Based Recommendations */}
        <Card style={[styles.sectionCard, styles.weatherCard]}>
          <Card.Content>
            <View style={styles.weatherHeader}>
              <MaterialCommunityIcons name="weather-partly-cloudy" size={24} color="#2196F3" />
              <Title style={[styles.sectionTitle, { marginBottom: 0, marginLeft: 8 }]}>
                Weather Recommendations
              </Title>
            </View>
            <Divider style={{ marginVertical: 12 }} />
            <View style={styles.recommendationsList}>
              <View style={styles.recommendationItem}>
                <MaterialCommunityIcons name="water" size={16} color="#2196F3" />
                <Text style={styles.recommendationText}>
                  Current conditions are favorable for field activities
                </Text>
              </View>
              <View style={styles.recommendationItem}>
                <MaterialCommunityIcons name="weather-sunny" size={16} color="#FF9800" />
                <Text style={styles.recommendationText}>
                  No rain expected for the next 3 days - perfect for harvesting
                </Text>
              </View>
              <View style={styles.recommendationItem}>
                <MaterialCommunityIcons name="thermometer" size={16} color="#4CAF50" />
                <Text style={styles.recommendationText}>
                  Optimal temperature range for field preparation activities
                </Text>
              </View>
            </View>
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
                        <MaterialCommunityIcons name="scissors-cutting" size={16} color="#FF9800" />
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
                      {crop.yield && (
                        <View style={styles.cropDetailRow}>
                          <MaterialCommunityIcons name="chart-line" size={16} color="#9C27B0" />
                          <Text style={styles.cropDetailText}>
                            Yield: {crop.yield}
                          </Text>
                        </View>
                      )}
                      {crop.marketPrice && (
                        <View style={styles.cropDetailRow}>
                          <MaterialCommunityIcons name="currency-inr" size={16} color="#4CAF50" />
                          <Text style={styles.cropDetailText}>
                            Price: {crop.marketPrice}
                          </Text>
                        </View>
                      )}
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
  // New styles for upcoming events
  eventsContainer: {
    marginTop: 8,
  },
  eventItem: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    padding: 12,
    marginBottom: 12,
    backgroundColor: '#FAFAFA',
    borderRadius: 8,
    borderLeftWidth: 3,
    borderLeftColor: '#4CAF50',
  },
  eventIcon: {
    width: 40,
    height: 40,
    borderRadius: 20,
    justifyContent: 'center',
    alignItems: 'center',
    marginRight: 12,
  },
  eventContent: {
    flex: 1,
  },
  eventHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 4,
  },
  eventTitle: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#333',
    flex: 1,
  },
  priorityBadge: {
    paddingHorizontal: 8,
    paddingVertical: 2,
    borderRadius: 12,
    marginLeft: 8,
  },
  priorityText: {
    fontSize: 10,
    color: 'white',
    fontWeight: 'bold',
  },
  eventDescription: {
    fontSize: 12,
    color: '#666',
    marginBottom: 8,
  },
  eventFooter: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  eventDate: {
    fontSize: 12,
    fontWeight: 'bold',
    color: '#4CAF50',
  },
  eventType: {
    fontSize: 10,
    color: '#999',
    textTransform: 'uppercase',
  },
  // Weather recommendations styles
  weatherCard: {
    backgroundColor: '#F0F8FF',
    borderColor: '#2196F3',
    borderWidth: 1,
  },
  weatherHeader: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  recommendationsList: {
    marginTop: 8,
  },
  recommendationItem: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 8,
    paddingVertical: 4,
  },
  recommendationText: {
    fontSize: 14,
    color: '#333',
    marginLeft: 8,
    flex: 1,
  },
});

export default CropCalendarScreen;
