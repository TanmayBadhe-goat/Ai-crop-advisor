import { Calendar, Bell, Droplets, Scissors, Sprout } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/card";
import { Badge } from "@/components/badge";

interface CalendarEvent {
  id: string;
  title: string;
  date: string;
  type: "sowing" | "irrigation" | "fertilizer" | "harvest";
  crop: string;
  description: string;
}

const CropCalendar = () => {
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
    <Card className="shadow-card">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Calendar className="h-5 w-5 text-primary" />
          Crop Calendar
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          <div className="text-sm text-muted-foreground mb-4">
            Personalized farming schedule based on your crops and location
          </div>
          
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
          
          <div className="mt-6 p-4 bg-gradient-earth rounded-lg">
            <h5 className="font-medium text-foreground mb-2">Weather-Based Recommendations</h5>
            <p className="text-sm text-muted-foreground">
              Current conditions are favorable for field activities. No rain expected for the next 3 days.
            </p>
          </div>
        </div>
      </CardContent>
    </Card>
  );
};

export default CropCalendar;