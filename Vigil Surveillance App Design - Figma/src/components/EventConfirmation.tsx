import { useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';
import { Button } from './ui/button';
import { Badge } from './ui/badge';
import { Textarea } from './ui/textarea';
import { Label } from './ui/label';
import { vigilClasses } from './vigil-theme';
import { toast } from 'sonner@2.0.3';
import {
  CheckCircle,
  XCircle,
  AlertTriangle,
  Clock,
  MapPin,
  Video,
  Flag,
  MessageSquare,
  User,
} from 'lucide-react';

interface PendingEvent {
  id: string;
  type: 'violence' | 'car-crash' | 'assault' | 'theft';
  location: string;
  camera: string;
  timestamp: string;
  confidence: number;
  status: 'pending' | 'confirmed' | 'dismissed' | 'false-alarm';
  thumbnailUrl?: string;
  aiNotes: string;
}

export function EventConfirmation() {
  const [events, setEvents] = useState<PendingEvent[]>([
    {
      id: '1',
      type: 'violence',
      location: 'Main Street & 5th Ave',
      camera: 'CAM-034',
      timestamp: '2 minutes ago',
      confidence: 94,
      status: 'pending',
      aiNotes: 'Detected aggressive behavior pattern with high confidence',
    },
    {
      id: '2',
      type: 'car-crash',
      location: 'Highway 101 Exit 23',
      camera: 'CAM-112',
      timestamp: '8 minutes ago',
      confidence: 89,
      status: 'pending',
      aiNotes: 'Vehicle collision detected with airbag deployment',
    },
    {
      id: '3',
      type: 'assault',
      location: 'Park Plaza North',
      camera: 'CAM-067',
      timestamp: '15 minutes ago',
      confidence: 87,
      status: 'pending',
      aiNotes: 'Physical altercation detected between multiple individuals',
    },
  ]);

  const [selectedEvent, setSelectedEvent] = useState<string | null>(null);
  const [annotationNotes, setAnnotationNotes] = useState<{ [key: string]: string }>({});

  const handleConfirmEvent = (eventId: string) => {
    setEvents(events.map(e => 
      e.id === eventId ? { ...e, status: 'confirmed' } : e
    ));
    toast.success('Event confirmed and escalated');
  };

  const handleDismissEvent = (eventId: string) => {
    setEvents(events.map(e => 
      e.id === eventId ? { ...e, status: 'dismissed' } : e
    ));
    toast.info('Event dismissed');
  };

  const handleFlagFalseAlarm = (eventId: string) => {
    setEvents(events.map(e => 
      e.id === eventId ? { ...e, status: 'false-alarm' } : e
    ));
    const notes = annotationNotes[eventId] || 'No notes provided';
    toast.success('Marked as false alarm - Data sent for model retraining');
  };

  const handleSaveAnnotation = (eventId: string) => {
    const notes = annotationNotes[eventId];
    if (!notes || notes.trim() === '') {
      toast.error('Please add annotation notes');
      return;
    }
    toast.success('Annotation saved for AI model retraining');
    setSelectedEvent(null);
  };

  const getTypeColor = (type: string) => {
    switch (type) {
      case 'violence':
        return 'text-red-600 dark:text-red-400 bg-red-100 dark:bg-red-900/30';
      case 'car-crash':
        return 'text-orange-600 dark:text-orange-400 bg-orange-100 dark:bg-orange-900/30';
      case 'assault':
        return 'text-purple-600 dark:text-purple-400 bg-purple-100 dark:bg-purple-900/30';
      case 'theft':
        return 'text-yellow-600 dark:text-yellow-400 bg-yellow-100 dark:bg-yellow-900/30';
      default:
        return 'text-gray-600 dark:text-gray-400 bg-gray-100 dark:bg-gray-900/30';
    }
  };

  const getStatusBadge = (status: string) => {
    switch (status) {
      case 'pending':
        return <Badge className={vigilClasses.badgeWarning}><Clock className="w-3 h-3 mr-1" />Pending Review</Badge>;
      case 'confirmed':
        return <Badge className={vigilClasses.badgeSuccess}><CheckCircle className="w-3 h-3 mr-1" />Confirmed</Badge>;
      case 'dismissed':
        return <Badge className="bg-gray-100 dark:bg-gray-900/30 text-gray-700 dark:text-gray-400"><XCircle className="w-3 h-3 mr-1" />Dismissed</Badge>;
      case 'false-alarm':
        return <Badge className={vigilClasses.badgeDanger}><Flag className="w-3 h-3 mr-1" />False Alarm</Badge>;
      default:
        return null;
    }
  };

  const pendingEvents = events.filter(e => e.status === 'pending');
  const reviewedEvents = events.filter(e => e.status !== 'pending');

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className={vigilClasses.textPrimary}>Event Confirmation</h1>
        <p className={vigilClasses.textSecondary}>
          Review and confirm AI-detected incidents
        </p>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card className={vigilClasses.card}>
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className={`text-sm ${vigilClasses.textMuted}`}>Pending Review</p>
                <p className={`text-2xl ${vigilClasses.textPrimary} mt-1`}>{pendingEvents.length}</p>
              </div>
              <div className="p-3 rounded-lg bg-amber-100 dark:bg-amber-900/30">
                <Clock className="w-6 h-6 text-amber-600 dark:text-amber-400" />
              </div>
            </div>
          </CardContent>
        </Card>

        <Card className={vigilClasses.card}>
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className={`text-sm ${vigilClasses.textMuted}`}>Confirmed Today</p>
                <p className={`text-2xl ${vigilClasses.textPrimary} mt-1`}>12</p>
              </div>
              <div className="p-3 rounded-lg bg-green-100 dark:bg-green-900/30">
                <CheckCircle className="w-6 h-6 text-green-600 dark:text-green-400" />
              </div>
            </div>
          </CardContent>
        </Card>

        <Card className={vigilClasses.card}>
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className={`text-sm ${vigilClasses.textMuted}`}>False Alarms</p>
                <p className={`text-2xl ${vigilClasses.textPrimary} mt-1`}>7</p>
              </div>
              <div className="p-3 rounded-lg bg-red-100 dark:bg-red-900/30">
                <Flag className="w-6 h-6 text-red-600 dark:text-red-400" />
              </div>
            </div>
          </CardContent>
        </Card>

        <Card className={vigilClasses.card}>
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className={`text-sm ${vigilClasses.textMuted}`}>Dismissed</p>
                <p className={`text-2xl ${vigilClasses.textPrimary} mt-1`}>5</p>
              </div>
              <div className="p-3 rounded-lg bg-gray-100 dark:bg-gray-900/30">
                <XCircle className="w-6 h-6 text-gray-600 dark:text-gray-400" />
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Pending Events */}
      <Card className={vigilClasses.card}>
        <CardHeader>
          <CardTitle className={vigilClasses.textPrimary}>Pending Events</CardTitle>
          <CardDescription>Events requiring immediate review and action</CardDescription>
        </CardHeader>
        <CardContent>
          {pendingEvents.length === 0 ? (
            <div className="text-center py-12">
              <CheckCircle className="w-12 h-12 text-green-500 mx-auto mb-3" />
              <p className={vigilClasses.textPrimary}>All events reviewed!</p>
              <p className={`text-sm ${vigilClasses.textMuted} mt-1`}>No pending events require your attention</p>
            </div>
          ) : (
            <div className="space-y-4">
              {pendingEvents.map((event) => (
                <div
                  key={event.id}
                  className={`p-4 rounded-lg border ${vigilClasses.border} ${vigilClasses.bgCardHover}`}
                >
                  <div className="flex flex-col lg:flex-row gap-4">
                    {/* Event Info */}
                    <div className="flex-1 space-y-3">
                      <div className="flex items-start justify-between">
                        <div className="flex items-center gap-3">
                          <div className={`p-2 rounded-lg ${getTypeColor(event.type)}`}>
                            <AlertTriangle className="w-5 h-5" />
                          </div>
                          <div>
                            <h4 className={vigilClasses.textPrimary}>
                              {event.type.split('-').map(word => word.charAt(0).toUpperCase() + word.slice(1)).join(' ')}
                            </h4>
                            <div className="flex items-center gap-3 mt-1">
                              <span className={`text-sm ${vigilClasses.textMuted} flex items-center gap-1`}>
                                <Clock className="w-3 h-3" />
                                {event.timestamp}
                              </span>
                              <span className={`text-sm ${vigilClasses.textMuted} flex items-center gap-1`}>
                                <Video className="w-3 h-3" />
                                {event.camera}
                              </span>
                            </div>
                          </div>
                        </div>
                        <Badge className={vigilClasses.badgeInfo}>{event.confidence}% Confidence</Badge>
                      </div>

                      <div className="flex items-start gap-2">
                        <MapPin className="w-4 h-4 text-blue-500 dark:text-cyan-500 mt-0.5" />
                        <span className={vigilClasses.textSecondary}>{event.location}</span>
                      </div>

                      <div className={`p-3 rounded-lg bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800`}>
                        <p className="text-sm text-blue-700 dark:text-blue-400">
                          <strong>AI Analysis:</strong> {event.aiNotes}
                        </p>
                      </div>

                      {/* Annotation Section */}
                      {selectedEvent === event.id && (
                        <div className="space-y-3 pt-2">
                          <Label htmlFor={`notes-${event.id}`} className={vigilClasses.textSecondary}>
                            Add annotation for retraining (optional)
                          </Label>
                          <Textarea
                            id={`notes-${event.id}`}
                            placeholder="Describe what the AI missed or got wrong..."
                            value={annotationNotes[event.id] || ''}
                            onChange={(e) => setAnnotationNotes({ ...annotationNotes, [event.id]: e.target.value })}
                            className={vigilClasses.input}
                            rows={3}
                          />
                          <Button
                            size="sm"
                            onClick={() => handleSaveAnnotation(event.id)}
                            className={vigilClasses.btnPrimary}
                          >
                            <MessageSquare className="w-4 h-4 mr-2" />
                            Save Annotation
                          </Button>
                        </div>
                      )}
                    </div>

                    {/* Actions */}
                    <div className="flex lg:flex-col gap-2 lg:min-w-[140px]">
                      <Button
                        size="sm"
                        onClick={() => handleConfirmEvent(event.id)}
                        className="flex-1 lg:flex-none bg-green-500 hover:bg-green-600 text-white"
                      >
                        <CheckCircle className="w-4 h-4 lg:mr-2" />
                        <span className="hidden lg:inline">Confirm</span>
                      </Button>
                      <Button
                        size="sm"
                        variant="outline"
                        onClick={() => handleDismissEvent(event.id)}
                        className={`flex-1 lg:flex-none ${vigilClasses.border}`}
                      >
                        <XCircle className="w-4 h-4 lg:mr-2" />
                        <span className="hidden lg:inline">Dismiss</span>
                      </Button>
                      <Button
                        size="sm"
                        variant="outline"
                        onClick={() => handleFlagFalseAlarm(event.id)}
                        className={`flex-1 lg:flex-none ${vigilClasses.border}`}
                      >
                        <Flag className="w-4 h-4 lg:mr-2" />
                        <span className="hidden lg:inline">False Alarm</span>
                      </Button>
                      <Button
                        size="sm"
                        variant="outline"
                        onClick={() => setSelectedEvent(selectedEvent === event.id ? null : event.id)}
                        className={`flex-1 lg:flex-none ${vigilClasses.border}`}
                      >
                        <MessageSquare className="w-4 h-4 lg:mr-2" />
                        <span className="hidden lg:inline">Annotate</span>
                      </Button>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </CardContent>
      </Card>

      {/* Reviewed Events */}
      {reviewedEvents.length > 0 && (
        <Card className={vigilClasses.card}>
          <CardHeader>
            <CardTitle className={vigilClasses.textPrimary}>Recently Reviewed</CardTitle>
            <CardDescription>Your recent event decisions</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {reviewedEvents.map((event) => (
                <div
                  key={event.id}
                  className={`flex items-center justify-between p-3 rounded-lg ${vigilClasses.bgCardHover} border ${vigilClasses.border}`}
                >
                  <div className="flex items-center gap-3">
                    <div className={`p-2 rounded-lg ${getTypeColor(event.type)}`}>
                      <AlertTriangle className="w-4 h-4" />
                    </div>
                    <div>
                      <p className={vigilClasses.textPrimary}>
                        {event.type.split('-').map(word => word.charAt(0).toUpperCase() + word.slice(1)).join(' ')} at {event.location}
                      </p>
                      <p className={`text-sm ${vigilClasses.textMuted}`}>{event.timestamp} • {event.camera}</p>
                    </div>
                  </div>
                  {getStatusBadge(event.status)}
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}