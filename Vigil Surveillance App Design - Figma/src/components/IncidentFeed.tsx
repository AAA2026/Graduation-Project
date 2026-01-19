import { useState } from 'react';
import { Card, CardContent } from './ui/card';
import { Button } from './ui/button';
import { Badge } from './ui/badge';
import { Input } from './ui/input';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './ui/select';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './ui/tabs';
import { ImageWithFallback } from './figma/ImageWithFallback';
import {
  AlertCircle,
  Car,
  Swords,
  MapPin,
  Clock,
  Eye,
  Play,
  CheckCircle,
  XCircle,
  Search,
  Filter,
  Shield,
  UserCheck,
  Video,
} from 'lucide-react';
import { toast } from 'sonner@2.0.3';
import { useTheme } from './ThemeProvider';
import { VideoModal } from './VideoModal';

interface IncidentFeedProps {
  role: 'admin' | 'officer' | 'security';
}

type IncidentStatus = 'pending_review' | 'confirmed' | 'dismissed' | 'resolved' | 'emergency_alert';
type IncidentType = 'violence' | 'crash';
type ConfidenceLevel = 'high' | 'medium' | 'low';

interface Incident {
  id: number;
  type: IncidentType;
  severity: 'critical' | 'high' | 'medium';
  location: string;
  camera: string;
  timestamp: string;
  duration: string;
  status: IncidentStatus;
  thumbnail: string;
  confidence: number;
  confidenceLevel: ConfidenceLevel;
  requiresReview: boolean;
  aiFeedback?: boolean;
  fieldResponse?: string;
}

export function IncidentFeed({ role }: IncidentFeedProps) {
  const [filter, setFilter] = useState('all');
  const [searchQuery, setSearchQuery] = useState('');
  const [incidents, setIncidents] = useState<Incident[]>([
    {
      id: 1,
      type: 'violence',
      severity: 'high',
      location: 'Main St & 5th Ave',
      camera: 'CAM-042',
      timestamp: '2024-11-04 14:32:15',
      duration: '45s',
      status: 'confirmed',
      thumbnail: 'fight street surveillance',
      confidence: 94,
      confidenceLevel: 'high',
      requiresReview: false,
      aiFeedback: true,
    },
    {
      id: 2,
      type: 'crash',
      severity: 'critical',
      location: 'Highway 101, Exit 23',
      camera: 'CAM-128',
      timestamp: '2024-11-04 14:29:42',
      duration: '1m 12s',
      status: 'pending_review',
      thumbnail: 'car accident highway',
      confidence: 72,
      confidenceLevel: 'medium',
      requiresReview: true,
    },
    {
      id: 3,
      type: 'violence',
      severity: 'medium',
      location: 'Park Plaza, North Wing',
      camera: 'CAM-089',
      timestamp: '2024-11-04 14:22:08',
      duration: '32s',
      status: 'pending_review',
      thumbnail: 'public plaza fight',
      confidence: 65,
      confidenceLevel: 'medium',
      requiresReview: true,
    },
    {
      id: 4,
      type: 'crash',
      severity: 'high',
      location: 'Downtown Bridge',
      camera: 'CAM-156',
      timestamp: '2024-11-04 14:16:33',
      duration: '56s',
      status: 'dismissed',
      thumbnail: 'city bridge traffic',
      confidence: 55,
      confidenceLevel: 'low',
      requiresReview: false,
      aiFeedback: false,
    },
    {
      id: 5,
      type: 'violence',
      severity: 'high',
      location: 'Shopping District, Block C',
      camera: 'CAM-203',
      timestamp: '2024-11-04 14:08:21',
      duration: '1m 5s',
      status: 'emergency_alert',
      thumbnail: 'shopping street assault',
      confidence: 92,
      confidenceLevel: 'high',
      requiresReview: false,
      aiFeedback: true,
      fieldResponse: 'Unit dispatched - ETA 5min',
    },
    {
      id: 6,
      type: 'crash',
      severity: 'medium',
      location: 'Riverside Parkway',
      camera: 'CAM-074',
      timestamp: '2024-11-04 13:55:47',
      duration: '41s',
      status: 'resolved',
      thumbnail: 'parkway road accident',
      confidence: 89,
      confidenceLevel: 'high',
      requiresReview: false,
    },
  ]);
  const { theme } = useTheme();
  const [selectedVideoIncident, setSelectedVideoIncident] = useState<Incident | null>(null);

  // Get confidence level based on score
  const getConfidenceLevel = (confidence: number): ConfidenceLevel => {
    if (confidence >= 85) return 'high';
    if (confidence >= 60) return 'medium';
    return 'low';
  };

  // Get confidence badge style
  const getConfidenceBadgeStyle = (level: ConfidenceLevel) => {
    switch (level) {
      case 'high':
        return 'bg-green-100 text-green-800 border-green-200 dark:bg-green-500/20 dark:text-green-400 dark:border-green-500/30';
      case 'medium':
        return 'bg-amber-100 text-amber-800 border-amber-200 dark:bg-amber-500/20 dark:text-amber-400 dark:border-amber-500/30';
      case 'low':
        return 'bg-gray-100 text-gray-800 border-gray-200 dark:bg-gray-500/20 dark:text-gray-400 dark:border-gray-500/30';
    }
  };

  // Handle officer confirming an incident
  const handleConfirmIncident = (id: number) => {
    setIncidents(prev => prev.map(incident => 
      incident.id === id 
        ? { 
            ...incident, 
            status: 'confirmed',
            requiresReview: false,
            aiFeedback: true
          }
        : incident
    ));
    
    // Trigger emergency alert for Security Authority
    triggerEmergencyAlert(id);
    toast.success('Incident confirmed - Emergency alert sent');
  };

  // Handle officer dismissing an incident
  const handleDismissIncident = (id: number) => {
    setIncidents(prev => prev.map(incident => 
      incident.id === id 
        ? { 
            ...incident, 
            status: 'dismissed',
            requiresReview: false,
            aiFeedback: false
          }
        : incident
    ));
    
    // Log false positive for AI retraining
    logFalsePositive(id);
    toast.info('Incident dismissed - False positive logged for AI training');
  };

  // Handle Security Authority acknowledging emergency alert
  const handleAcknowledgeEmergency = (id: number) => {
    setIncidents(prev => prev.map(incident => 
      incident.id === id 
        ? { 
            ...incident, 
            status: 'confirmed',
            fieldResponse: 'Acknowledged - Response initiated'
          }
        : incident
    ));
    toast.success('Emergency alert acknowledged');
  };

  // Handle Security Authority sending field response
  const handleSendFieldResponse = (id: number) => {
    setIncidents(prev => prev.map(incident => 
      incident.id === id 
        ? { 
            ...incident, 
            status: 'resolved',
            fieldResponse: 'Field unit deployed - Situation contained'
          }
        : incident
    ));
    toast.success('Field response report sent');
  };

  // Trigger emergency alert (simulated)
  const triggerEmergencyAlert = (incidentId: number) => {
    console.log(`🚨 EMERGENCY ALERT: Incident ${incidentId} requires immediate attention`);
    // In real implementation, this would call your notification service
  };

  // Log false positive for AI retraining (simulated)
  const logFalsePositive = (incidentId: number) => {
    console.log(`📝 AI FEEDBACK: False positive logged for incident ${incidentId}`);
    // In real implementation, this would update AI training dataset
  };

  const filteredIncidents = incidents.filter((incident) => {
    if (filter !== 'all' && incident.type !== filter) return false;
    if (
      searchQuery &&
      !incident.location.toLowerCase().includes(searchQuery.toLowerCase()) &&
      !incident.camera.toLowerCase().includes(searchQuery.toLowerCase())
    ) {
      return false;
    }
    return true;
  });

  // Get status badge style
  const getStatusBadgeStyle = (status: IncidentStatus) => {
    switch (status) {
      case 'pending_review':
        return 'border-amber-500/50 text-amber-600 dark:text-amber-400 bg-amber-100 dark:bg-amber-500/10';
      case 'confirmed':
        return 'border-blue-500/50 text-blue-600 dark:text-blue-400 bg-blue-100 dark:bg-blue-500/10';
      case 'emergency_alert':
        return 'border-red-500/50 text-red-600 dark:text-red-400 bg-red-100 dark:bg-red-500/10 animate-pulse';
      case 'dismissed':
        return 'border-gray-500/50 text-gray-600 dark:text-gray-400 bg-gray-100 dark:bg-gray-500/10';
      case 'resolved':
        return 'border-green-500/50 text-green-600 dark:text-green-400 bg-green-100 dark:bg-green-500/10';
    }
  };

  // Get status display text
  const getStatusText = (status: IncidentStatus) => {
    switch (status) {
      case 'pending_review': return 'Review Needed';
      case 'confirmed': return 'Confirmed';
      case 'emergency_alert': return 'Emergency Alert';
      case 'dismissed': return 'Dismissed';
      case 'resolved': return 'Resolved';
    }
  };

  return (
    <div className="space-y-6">
      {/* Video Modal */}
      {selectedVideoIncident && (
        <VideoModal
          isOpen={!!selectedVideoIncident}
          onClose={() => setSelectedVideoIncident(null)}
          videoUrl={`http://127.0.0.1:5000/videos/${selectedVideoIncident.camera}_sample.mp4`}
          cameraId={selectedVideoIncident.camera}
          timestamp={selectedVideoIncident.timestamp}
          incidentType={selectedVideoIncident.type}
          location={selectedVideoIncident.location}
          showViewLiveButton={true}
        />
      )}

      {/* Header */}
      <div className="flex flex-col gap-3 sm:gap-4 md:flex-row md:items-center md:justify-between">
        <div>
          <h2 className="text-xl sm:text-2xl font-bold text-foreground mb-1">
            {role === 'admin' ? 'Incident Feed' : role === 'security' ? 'Emergency Alerts' : 'Live Alerts'}
          </h2>
          <p className="text-sm sm:text-base text-muted-foreground">
            {role === 'security' 
              ? 'Critical incidents requiring immediate attention' 
              : 'Real-time incident detection and monitoring'
            }
          </p>
        </div>
        <div className="flex gap-2 flex-wrap">
          <Badge variant="destructive" className="animate-pulse text-xs sm:text-sm">
            <div className="w-2 h-2 rounded-full bg-white mr-2" />
            Live
          </Badge>
          {role === 'officer' && (
            <Badge variant="outline" className="border-amber-500/50 text-amber-600 dark:text-amber-400 text-xs sm:text-sm">
              <UserCheck className="w-3 h-3 mr-1" />
              {incidents.filter(i => i.status === 'pending_review').length} Pending
            </Badge>
          )}
          {role === 'security' && (
            <Badge variant="destructive" className="text-xs sm:text-sm">
              <Shield className="w-3 h-3 mr-1" />
              {incidents.filter(i => i.status === 'emergency_alert').length} Emergency
            </Badge>
          )}
        </div>
      </div>

      {/* Filters */}
      <Card className="bg-card/80 backdrop-blur border-border">
        <CardContent className="p-3 sm:p-4">
          <div className="flex flex-col gap-2 sm:gap-3 md:flex-row md:items-center">
            <div className="flex-1 relative">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
              <Input
                placeholder="Search by location or camera..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="pl-10 bg-background border-border text-foreground placeholder:text-muted-foreground text-sm sm:text-base h-9 sm:h-10"
              />
            </div>
            <Select value={filter} onValueChange={setFilter}>
              <SelectTrigger className="w-full md:w-48 bg-background border-border text-foreground text-sm sm:text-base h-9 sm:h-10">
                <Filter className="w-4 h-4 mr-2" />
                <SelectValue />
              </SelectTrigger>
              <SelectContent className="bg-card border-border text-foreground">
                <SelectItem value="all">All Types</SelectItem>
                <SelectItem value="violence">Violence Only</SelectItem>
                <SelectItem value="crash">Crashes Only</SelectItem>
              </SelectContent>
            </Select>
            {/* Confidence Filter for Officers */}
            {role === 'officer' && (
              <Select onValueChange={(value) => {
                if (value === 'all') setFilter('all');
                else setFilter(value);
              }}>
                <SelectTrigger className="w-full md:w-48 bg-background border-border text-foreground text-sm sm:text-base h-9 sm:h-10">
                  <AlertCircle className="w-4 h-4 mr-2" />
                  <SelectValue placeholder="Confidence" />
                </SelectTrigger>
                <SelectContent className="bg-card border-border text-foreground">
                  <SelectItem value="all">All Confidence</SelectItem>
                  <SelectItem value="high">High Confidence</SelectItem>
                  <SelectItem value="medium">Medium Confidence</SelectItem>
                  <SelectItem value="low">Low Confidence</SelectItem>
                </SelectContent>
              </Select>
            )}
          </div>
        </CardContent>
      </Card>

      {/* Incident Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-3 sm:gap-4">
        {filteredIncidents.map((incident) => (
          <Card
            key={incident.id}
            className={`bg-card/80 backdrop-blur border-border overflow-hidden transition-all ${
              incident.status === 'emergency_alert' 
                ? 'border-red-500/50 shadow-lg shadow-red-500/20' 
                : incident.requiresReview 
                ? 'border-amber-500/30 hover:border-amber-500/50' 
                : 'hover:border-blue-500/50'
            }`}
          >
            <div className="relative aspect-video bg-muted">
              <ImageWithFallback
                src={`https://source.unsplash.com/800x450/?${incident.thumbnail}`}
                alt={incident.type}
                className="w-full h-full object-cover"
              />
              <div className="absolute inset-0 bg-gradient-to-t from-black/60 to-transparent" />
              
              {/* Overlay Badges */}
              <div className="absolute top-3 left-3 flex gap-2">
                <Badge
                  className={`${
                    incident.type === 'violence'
                      ? 'bg-red-500/90 text-white border-0'
                      : 'bg-amber-500/90 text-white border-0'
                  }`}
                >
                  {incident.type === 'violence' ? (
                    <Swords className="w-3 h-3 mr-1" />
                  ) : (
                    <Car className="w-3 h-3 mr-1" />
                  )}
                  {incident.type === 'violence' ? 'Violence' : 'Car Crash'}
                </Badge>
                <Badge className={getConfidenceBadgeStyle(incident.confidenceLevel)}>
                  {incident.confidence}% {incident.confidenceLevel}
                </Badge>
              </div>

              {/* AI Feedback Indicator */}
              {incident.aiFeedback !== undefined && (
                <div className="absolute top-3 right-3">
                  <Badge className={`${
                    incident.aiFeedback 
                      ? 'bg-green-500/90 text-white border-0' 
                      : 'bg-gray-500/90 text-white border-0'
                  }`}>
                    AI: {incident.aiFeedback ? '✓' : '✗'}
                  </Badge>
                </div>
              )}

              {/* Play Button */}
              <button className="absolute inset-0 flex items-center justify-center group">
                <div className="w-16 h-16 rounded-full bg-white/20 backdrop-blur flex items-center justify-center group-hover:bg-white/30 transition-colors">
                  <Play className="w-8 h-8 text-white ml-1" />
                </div>
              </button>

              {/* Duration */}
              <div className="absolute bottom-3 right-3">
                <Badge className="bg-black/60 text-white border-0">
                  <Clock className="w-3 h-3 mr-1" />
                  {incident.duration}
                </Badge>
              </div>
            </div>

            <CardContent className="p-3 sm:p-4 space-y-2 sm:space-y-3">
              {/* Location and Camera */}
              <div className="space-y-1 sm:space-y-2">
                <div className="flex items-start gap-2 text-foreground text-sm sm:text-base">
                  <MapPin className="w-4 h-4 mt-0.5 text-blue-600 dark:text-blue-400 flex-shrink-0" />
                  <span className="break-words">{incident.location}</span>
                </div>
                <div className="flex flex-col sm:flex-row sm:items-center gap-1 sm:gap-4 text-xs sm:text-sm text-muted-foreground">
                  <span className="flex items-center gap-1">
                    <Eye className="w-3 h-3" />
                    {incident.camera}
                  </span>
                  <span className="truncate">{incident.timestamp}</span>
                </div>
              </div>

              {/* Status and Field Response */}
              <div className="flex items-center justify-between gap-2 flex-wrap">
                <Badge variant="outline" className={`${getStatusBadgeStyle(incident.status)} text-xs`}>
                  {getStatusText(incident.status)}
                </Badge>
                {incident.fieldResponse && (
                  <Badge variant="outline" className="border-green-500/50 text-green-600 dark:text-green-400 text-xs flex-shrink-0">
                    {incident.fieldResponse}
                  </Badge>
                )}
              </div>

              {/* Action Buttons Based on Role and Status */}
              <div className="flex gap-2 pt-1 sm:pt-2 flex-wrap">
                {/* View Video Button - Always Available */}
                <Button
                  size="sm"
                  variant="outline"
                  className="border-cyan-500/50 text-cyan-600 dark:text-cyan-400 hover:bg-cyan-500/10 text-xs sm:text-sm h-8 sm:h-9"
                  onClick={() => setSelectedVideoIncident(incident)}
                >
                  <Video className="w-3 h-3 sm:w-4 sm:h-4 mr-1" />
                  View Video
                </Button>

                {/* Officer Actions */}
                {role === 'officer' && incident.status === 'pending_review' && (
                  <>
                    <Button
                      size="sm"
                      variant="outline"
                      className="flex-1 border-border text-foreground hover:bg-muted text-xs sm:text-sm h-8 sm:h-9"
                      onClick={() => handleDismissIncident(incident.id)}
                    >
                      <XCircle className="w-3 h-3 sm:w-4 sm:h-4 mr-1" />
                      Dismiss
                    </Button>
                    <Button
                      size="sm"
                      className="flex-1 bg-blue-600 hover:bg-blue-700 text-white text-xs sm:text-sm h-8 sm:h-9"
                      onClick={() => handleConfirmIncident(incident.id)}
                    >
                      <CheckCircle className="w-3 h-3 sm:w-4 sm:h-4 mr-1" />
                      Confirm
                    </Button>
                  </>
                )}

                {/* Security Authority Actions */}
                {role === 'security' && incident.status === 'emergency_alert' && (
                  <>
                    <Button
                      size="sm"
                      variant="outline"
                      className="flex-1 border-border text-foreground hover:bg-muted text-xs sm:text-sm h-8 sm:h-9"
                      onClick={() => handleAcknowledgeEmergency(incident.id)}
                    >
                      <Shield className="w-3 h-3 sm:w-4 sm:h-4 mr-1" />
                      <span className="hidden sm:inline">Acknowledge</span>
                      <span className="sm:hidden">ACK</span>
                    </Button>
                    <Button
                      size="sm"
                      className="flex-1 bg-green-600 hover:bg-green-700 text-white text-xs sm:text-sm h-8 sm:h-9"
                      onClick={() => handleSendFieldResponse(incident.id)}
                    >
                      <UserCheck className="w-3 h-3 sm:w-4 sm:h-4 mr-1" />
                      <span className="hidden sm:inline">Field Response</span>
                      <span className="sm:hidden">Respond</span>
                    </Button>
                  </>
                )}

                {/* Admin Actions */}
                {role === 'admin' && incident.status === 'confirmed' && (
                  <Button
                    size="sm"
                    className="flex-1 bg-green-600 hover:bg-green-700 text-white text-xs sm:text-sm h-8 sm:h-9"
                    onClick={() => handleSendFieldResponse(incident.id)}
                  >
                    <CheckCircle className="w-3 h-3 sm:w-4 sm:h-4 mr-1" />
                    Mark Resolved
                  </Button>
                )}
              </div>
            </CardContent>
          </Card>
        ))}
      </div>
    </div>
  );
}