import { Calendar, Bell, Droplets, Scissors, Sprout, ChevronLeft, ChevronRight, Filter } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/card";
import { Badge } from "@/components/badge";
import { Button } from "@/components/button";
import { useState } from "react";

interface CalendarEvent {
  id: string;
  title: string;
  date: string;
  type: "sowing" | "irrigation" | "fertilizer" | "harvest";
  crop: string;
  description: string;
}

interface CropSeason {
  crop: string;
  emoji: string;
  season: string;
  plantingMonths: string[];
  harvestMonths: string[];
  duration: string;
  tips: string;
  color: string;
  yield: string;
  marketPrice: string;
}

const CropCalendar = () => {
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
  const getEventIcon = (type: string) => {
    switch (type) {
      case "sowing": return <Sprout className="h-4 w-4" />;
      case "irrigation": return <Droplets className="h-4 w-4" />;
      case "fertilizer": return <Bell className="h-4 w-4" />;
      case "harvest": return <Scissors className="h-4 w-4" />;
      default: return <Calendar className="h-4 w-4" />;
    }
  };

  const getEventColor = (type: string) => {
    switch (type) {
      case "sowing": return "bg-success/20 text-success border-success/30";
      case "irrigation": return "bg-blue-500/20 text-blue-600 border-blue-500/30";
      case "fertilizer": return "bg-warning/20 text-warning border-warning/30";
      case "harvest": return "bg-crop-grain/20 text-crop-grain border-crop-grain/30";
      default: return "bg-muted text-muted-foreground";
    }
  };

  const upcomingEvents: CalendarEvent[] = [
    {
      id: "1",
      title: "Rice Transplanting",
      date: "Tomorrow",
      type: "sowing",
      crop: "Rice",
      description: "Optimal time for rice transplanting in Kharif season"
    },
    {
      id: "2",
      title: "Wheat Irrigation",
      date: "In 3 days",
      type: "irrigation",
      crop: "Wheat",
      description: "Second irrigation recommended for wheat crop"
    },
    {
      id: "3",
      title: "Cotton Fertilizer",
      date: "In 5 days",
      type: "fertilizer",
      crop: "Cotton",
      description: "Apply NPK fertilizer for cotton flowering stage"
    },
    {
      id: "4",
      title: "Tomato Harvest",
      date: "In 1 week",
      type: "harvest",
      crop: "Tomato",
      description: "First harvest of tomato crop ready"
    }
  ];

  return (
    <div className="space-y-6">
      {/* Month Selector */}
      <Card className="shadow-card">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Calendar className="h-5 w-5 text-primary" />
            Crop Calendar - {months[selectedMonth]} {new Date().getFullYear()}
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex items-center justify-between mb-4">
            <Button
              variant="outline"
              size="sm"
              onClick={() => setSelectedMonth(selectedMonth === 0 ? 11 : selectedMonth - 1)}
            >
              <ChevronLeft className="h-4 w-4" />
            </Button>
            <div className="flex gap-2 overflow-x-auto">
              {months.map((month, index) => (
                <Button
                  key={month}
                  variant={selectedMonth === index ? "default" : "outline"}
                  size="sm"
                  onClick={() => setSelectedMonth(index)}
                  className="min-w-[60px]"
                >
                  {month}
                </Button>
              ))}
            </div>
            <Button
              variant="outline"
              size="sm"
              onClick={() => setSelectedMonth(selectedMonth === 11 ? 0 : selectedMonth + 1)}
            >
              <ChevronRight className="h-4 w-4" />
            </Button>
          </div>

          {/* Season Filter */}
          <div className="flex items-center gap-2 mb-4">
            <Filter className="h-4 w-4 text-muted-foreground" />
            <span className="text-sm font-medium">Filter by Season:</span>
            {seasons.map((season) => (
              <Button
                key={season}
                variant={selectedSeason === season ? "default" : "outline"}
                size="sm"
                onClick={() => setSelectedSeason(season as any)}
              >
                {season}
              </Button>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Current Month Activities */}
      <Card className="shadow-card">
        <CardHeader>
          <CardTitle>Activities for {months[selectedMonth]}</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid md:grid-cols-2 gap-4">
            {/* Planting Activities */}
            {cropsToPlant.length > 0 && (
              <div className="space-y-3">
                <div className="flex items-center gap-2">
                  <Sprout className="h-5 w-5 text-green-600" />
                  <h4 className="font-medium text-green-600">Time to Plant</h4>
                </div>
                <div className="space-y-2">
                  {cropsToPlant.map((crop, index) => (
                    <div key={index} className="flex items-center gap-3 p-3 bg-green-50 rounded-lg border border-green-200">
                      <span className="text-2xl">{crop.emoji}</span>
                      <div>
                        <p className="font-medium">{crop.crop}</p>
                        <p className="text-sm text-muted-foreground">{crop.duration}</p>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Harvesting Activities */}
            {cropsToHarvest.length > 0 && (
              <div className="space-y-3">
                <div className="flex items-center gap-2">
                  <Scissors className="h-5 w-5 text-orange-600" />
                  <h4 className="font-medium text-orange-600">Time to Harvest</h4>
                </div>
                <div className="space-y-2">
                  {cropsToHarvest.map((crop, index) => (
                    <div key={index} className="flex items-center gap-3 p-3 bg-orange-50 rounded-lg border border-orange-200">
                      <span className="text-2xl">{crop.emoji}</span>
                      <div>
                        <p className="font-medium">{crop.crop}</p>
                        <p className="text-sm text-muted-foreground">{crop.duration}</p>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {cropsToPlant.length === 0 && cropsToHarvest.length === 0 && (
              <div className="col-span-2 text-center py-8">
                <Calendar className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
                <p className="text-muted-foreground">
                  No major planting or harvesting activities for {months[selectedMonth]}
                </p>
              </div>
            )}
          </div>
        </CardContent>
      </Card>

      {/* Upcoming Events */}
      <Card className="shadow-card">
        <CardHeader>
          <CardTitle>Upcoming Events</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-3">
            {upcomingEvents.map((event) => (
              <div
                key={event.id}
                className="flex items-center gap-3 p-3 rounded-lg border bg-card hover:shadow-soft transition-shadow"
              >
                <div className={`p-2 rounded-lg ${getEventColor(event.type)}`}>
                  {getEventIcon(event.type)}
                </div>
                
                <div className="flex-1">
                  <div className="flex items-center gap-2 mb-1">
                    <h4 className="font-medium text-foreground">{event.title}</h4>
                    <Badge variant="outline" className="text-xs">
                      {event.crop}
                    </Badge>
                  </div>
                  <p className="text-sm text-muted-foreground">{event.description}</p>
                </div>
                
                <div className="text-right">
                  <p className="text-sm font-medium text-primary">{event.date}</p>
                  <p className="text-xs text-muted-foreground capitalize">{event.type}</p>
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Crop Calendar Overview */}
      <Card className="shadow-card">
        <CardHeader>
          <CardTitle>Crop Calendar Overview</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid gap-4">
            {getFilteredCrops().map((crop, index) => (
              <Card key={index} className="border-l-4 hover:shadow-md transition-shadow" style={{ borderLeftColor: crop.color }}>
                <CardContent className="p-4">
                  <div className="flex items-start gap-4">
                    <span className="text-3xl">{crop.emoji}</span>
                    <div className="flex-1">
                      <div className="flex items-center gap-2 mb-2">
                        <h4 className="font-bold text-lg">{crop.crop}</h4>
                        <Badge variant="secondary">{crop.season}</Badge>
                      </div>
                      
                      <div className="grid md:grid-cols-2 gap-4 mb-3">
                        <div className="space-y-2">
                          <div className="flex items-center gap-2">
                            <Sprout className="h-4 w-4 text-green-600" />
                            <span className="text-sm">Plant: {crop.plantingMonths.join(', ')}</span>
                          </div>
                          <div className="flex items-center gap-2">
                            <Scissors className="h-4 w-4 text-orange-600" />
                            <span className="text-sm">Harvest: {crop.harvestMonths.join(', ')}</span>
                          </div>
                        </div>
                        <div className="space-y-2">
                          <div className="flex items-center gap-2">
                            <Calendar className="h-4 w-4 text-blue-600" />
                            <span className="text-sm">Duration: {crop.duration}</span>
                          </div>
                          <div className="flex items-center gap-2">
                            <span className="text-sm">💰 {crop.marketPrice}</span>
                          </div>
                        </div>
                      </div>
                      
                      <p className="text-sm text-muted-foreground bg-muted/50 p-2 rounded italic">
                        {crop.tips}
                      </p>
                    </div>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Weather-Based Recommendations */}
      <Card className="shadow-card">
        <CardContent className="p-4 bg-gradient-to-r from-blue-50 to-green-50">
          <h5 className="font-medium text-foreground mb-2 flex items-center gap-2">
            <Droplets className="h-4 w-4" />
            Weather-Based Recommendations
          </h5>
          <p className="text-sm text-muted-foreground">
            Current conditions are favorable for field activities. No rain expected for the next 3 days.
            Perfect time for harvesting and field preparation activities.
          </p>
        </CardContent>
      </Card>
    </div>
  );
};

export default CropCalendar;