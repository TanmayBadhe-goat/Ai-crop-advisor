import React, { useState } from 'react';
import { View, StyleSheet, ScrollView, TouchableOpacity, Dimensions } from 'react-native';
import { Card, Title, Paragraph, Text, Button, Avatar, Chip, FAB } from 'react-native-paper';
import { MaterialCommunityIcons } from '@expo/vector-icons';

const { width } = Dimensions.get('window');

interface Post {
  id: string;
  author: string;
  avatar: string;
  location: string;
  timeAgo: string;
  content: string;
  image?: string;
  likes: number;
  comments: number;
  tags: string[];
  category: 'Question' | 'Success Story' | 'Tip' | 'Alert';
}

const CommunityScreen = ({ navigation }: any) => {
  const [selectedCategory, setSelectedCategory] = useState<'All' | 'Question' | 'Success Story' | 'Tip' | 'Alert'>('All');

  const categories = ['All', 'Question', 'Success Story', 'Tip', 'Alert'];

  const samplePosts: Post[] = [
    {
      id: '1',
      author: 'Rajesh Kumar',
      avatar: '👨‍🌾',
      location: 'Punjab, India',
      timeAgo: '2 hours ago',
      content: 'Just harvested my wheat crop using the recommendations from CropAI! Got 15% better yield than last year. The weather alerts really helped me time the harvest perfectly.',
      likes: 24,
      comments: 8,
      tags: ['wheat', 'harvest', 'success'],
      category: 'Success Story'
    },
    {
      id: '2',
      author: 'Priya Sharma',
      avatar: '👩‍🌾',
      location: 'Maharashtra, India',
      timeAgo: '4 hours ago',
      content: 'Has anyone dealt with leaf blight in tomatoes? I\'m seeing some yellowing on my plants. The disease detection suggested it might be early blight. What treatments have worked for you?',
      likes: 12,
      comments: 15,
      tags: ['tomato', 'disease', 'help-needed'],
      category: 'Question'
    },
    {
      id: '3',
      author: 'Agricultural Expert',
      avatar: '🎓',
      location: 'IARI, Delhi',
      timeAgo: '6 hours ago',
      content: 'Pro Tip: Before the monsoon season, ensure your drainage systems are clean. Waterlogging can cause root rot in many crops. Also, consider companion planting with marigolds to naturally repel pests.',
      likes: 45,
      comments: 12,
      tags: ['monsoon', 'drainage', 'companion-planting'],
      category: 'Tip'
    },
    {
      id: '4',
      author: 'Weather Alert System',
      avatar: '🌦️',
      location: 'Karnataka, India',
      timeAgo: '8 hours ago',
      content: '⚠️ Heavy rainfall expected in Karnataka region for next 3 days. Farmers are advised to postpone harvesting activities and ensure proper drainage in fields. Stay safe!',
      likes: 67,
      comments: 23,
      tags: ['weather-alert', 'karnataka', 'rainfall'],
      category: 'Alert'
    },
    {
      id: '5',
      author: 'Amit Patel',
      avatar: '👨‍🌾',
      location: 'Gujarat, India',
      timeAgo: '1 day ago',
      content: 'Started using drip irrigation system this season. Water consumption reduced by 40% and crop quality improved significantly. Initial investment was worth it!',
      likes: 38,
      comments: 19,
      tags: ['irrigation', 'water-saving', 'technology'],
      category: 'Success Story'
    }
  ];

  const getFilteredPosts = () => {
    if (selectedCategory === 'All') {
      return samplePosts;
    }
    return samplePosts.filter(post => post.category === selectedCategory);
  };

  const getCategoryColor = (category: string) => {
    switch (category) {
      case 'Question': return '#2196F3';
      case 'Success Story': return '#4CAF50';
      case 'Tip': return '#FF9800';
      case 'Alert': return '#F44336';
      default: return '#666';
    }
  };

  const getCategoryIcon = (category: string) => {
    switch (category) {
      case 'Question': return 'help-circle';
      case 'Success Story': return 'trophy';
      case 'Tip': return 'lightbulb';
      case 'Alert': return 'alert';
      default: return 'forum';
    }
  };

  return (
    <View style={styles.container}>
      <ScrollView showsVerticalScrollIndicator={false}>
        <View style={styles.content}>
          <Title style={styles.title}>Community</Title>
          <Paragraph style={styles.subtitle}>
            Connect with fellow farmers and agricultural experts
          </Paragraph>

          {/* Category Filter */}
          <Card style={styles.filterCard}>
            <Card.Content>
              <ScrollView horizontal showsHorizontalScrollIndicator={false}>
                <View style={styles.categoryContainer}>
                  {categories.map((category) => (
                    <Chip
                      key={category}
                      selected={selectedCategory === category}
                      onPress={() => setSelectedCategory(category as any)}
                      style={[
                        styles.categoryChip,
                        selectedCategory === category && { backgroundColor: '#4CAF50' }
                      ]}
                      textStyle={[
                        styles.categoryText,
                        selectedCategory === category && { color: 'white' }
                      ]}
                    >
                      {category}
                    </Chip>
                  ))}
                </View>
              </ScrollView>
            </Card.Content>
          </Card>

          {/* Coming Soon Notice */}
          <Card style={styles.comingSoonCard}>
            <Card.Content style={styles.comingSoonContent}>
              <MaterialCommunityIcons name="hammer-wrench" size={48} color="#FF9800" />
              <Title style={styles.comingSoonTitle}>Community Feature Coming Soon!</Title>
              <Paragraph style={styles.comingSoonText}>
                We're building an amazing community platform where farmers can:
              </Paragraph>
              <View style={styles.featureList}>
                <Text style={styles.featureItem}>• Share farming experiences and success stories</Text>
                <Text style={styles.featureItem}>• Ask questions and get expert advice</Text>
                <Text style={styles.featureItem}>• Receive real-time weather and crop alerts</Text>
                <Text style={styles.featureItem}>• Connect with local farmers in your area</Text>
                <Text style={styles.featureItem}>• Access market prices and trends</Text>
              </View>
              <Button
                mode="contained"
                onPress={() => navigation.navigate('Dashboard')}
                style={styles.backButton}
                icon="home"
              >
                Back to Dashboard
              </Button>
            </Card.Content>
          </Card>

          {/* Sample Posts Preview */}
          <Card style={styles.sectionCard}>
            <Card.Content>
              <Title style={styles.sectionTitle}>Preview: Community Posts</Title>
              <Paragraph style={styles.sectionSubtitle}>
                Here's what the community will look like:
              </Paragraph>
              
              {getFilteredPosts().slice(0, 3).map((post) => (
                <Card key={post.id} style={styles.postCard}>
                  <Card.Content>
                    <View style={styles.postHeader}>
                      <View style={styles.authorInfo}>
                        <Text style={styles.authorAvatar}>{post.avatar}</Text>
                        <View style={styles.authorDetails}>
                          <Text style={styles.authorName}>{post.author}</Text>
                          <Text style={styles.postLocation}>{post.location} • {post.timeAgo}</Text>
                        </View>
                      </View>
                      <Chip
                        style={[styles.categoryBadge, { backgroundColor: getCategoryColor(post.category) }]}
                        textStyle={styles.categoryBadgeText}
                      >
                        {post.category}
                      </Chip>
                    </View>
                    
                    <Text style={styles.postContent}>{post.content}</Text>
                    
                    <View style={styles.postTags}>
                      {post.tags.map((tag, index) => (
                        <Chip key={index} style={styles.tagChip} textStyle={styles.tagText}>
                          #{tag}
                        </Chip>
                      ))}
                    </View>
                    
                    <View style={styles.postActions}>
                      <TouchableOpacity style={styles.actionButton}>
                        <MaterialCommunityIcons name="heart-outline" size={20} color="#666" />
                        <Text style={styles.actionText}>{post.likes}</Text>
                      </TouchableOpacity>
                      <TouchableOpacity style={styles.actionButton}>
                        <MaterialCommunityIcons name="comment-outline" size={20} color="#666" />
                        <Text style={styles.actionText}>{post.comments}</Text>
                      </TouchableOpacity>
                      <TouchableOpacity style={styles.actionButton}>
                        <MaterialCommunityIcons name="share-outline" size={20} color="#666" />
                        <Text style={styles.actionText}>Share</Text>
                      </TouchableOpacity>
                    </View>
                  </Card.Content>
                </Card>
              ))}
            </Card.Content>
          </Card>

          <View style={styles.bottomSpacing} />
        </View>
      </ScrollView>

      {/* Floating Action Button */}
      <FAB
        style={styles.fab}
        icon="plus"
        onPress={() => {
          // This would open a "Create Post" screen in the future
        }}
        disabled
      />
    </View>
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
  filterCard: {
    marginBottom: 16,
    elevation: 2,
    borderRadius: 12,
  },
  categoryContainer: {
    flexDirection: 'row',
    paddingVertical: 8,
  },
  categoryChip: {
    marginRight: 8,
  },
  categoryText: {
    fontSize: 12,
  },
  comingSoonCard: {
    marginBottom: 20,
    elevation: 3,
    borderRadius: 12,
    backgroundColor: '#FFF3E0',
  },
  comingSoonContent: {
    alignItems: 'center',
    paddingVertical: 20,
  },
  comingSoonTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#FF9800',
    marginTop: 12,
    marginBottom: 8,
    textAlign: 'center',
  },
  comingSoonText: {
    fontSize: 14,
    color: '#666',
    textAlign: 'center',
    marginBottom: 16,
  },
  featureList: {
    alignSelf: 'stretch',
    marginBottom: 20,
  },
  featureItem: {
    fontSize: 14,
    color: '#555',
    marginBottom: 8,
    paddingLeft: 8,
  },
  backButton: {
    backgroundColor: '#4CAF50',
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
    marginBottom: 4,
  },
  sectionSubtitle: {
    fontSize: 14,
    color: '#666',
    marginBottom: 16,
  },
  postCard: {
    marginBottom: 12,
    elevation: 1,
    borderRadius: 8,
    backgroundColor: '#FAFAFA',
  },
  postHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'flex-start',
    marginBottom: 12,
  },
  authorInfo: {
    flexDirection: 'row',
    alignItems: 'center',
    flex: 1,
  },
  authorAvatar: {
    fontSize: 32,
    marginRight: 12,
  },
  authorDetails: {
    flex: 1,
  },
  authorName: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#333',
  },
  postLocation: {
    fontSize: 12,
    color: '#666',
    marginTop: 2,
  },
  categoryBadge: {
    paddingHorizontal: 8,
    height: 24,
  },
  categoryBadgeText: {
    color: 'white',
    fontSize: 10,
    fontWeight: 'bold',
  },
  postContent: {
    fontSize: 14,
    color: '#333',
    lineHeight: 20,
    marginBottom: 12,
  },
  postTags: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    marginBottom: 12,
  },
  tagChip: {
    marginRight: 6,
    marginBottom: 4,
    backgroundColor: '#E8F5E9',
    height: 24,
  },
  tagText: {
    fontSize: 10,
    color: '#4CAF50',
  },
  postActions: {
    flexDirection: 'row',
    justifyContent: 'space-around',
    paddingTop: 8,
    borderTopWidth: 1,
    borderTopColor: '#E0E0E0',
  },
  actionButton: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: 4,
  },
  actionText: {
    fontSize: 12,
    color: '#666',
    marginLeft: 4,
  },
  fab: {
    position: 'absolute',
    margin: 16,
    right: 0,
    bottom: 0,
    backgroundColor: '#4CAF50',
  },
  bottomSpacing: {
    height: 80,
  },
});

export default CommunityScreen;
