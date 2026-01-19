import { useState, useCallback } from 'react';
import { useIncidentWebSocket } from '../hooks/useIncidentWebSocket';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Button } from './ui/button';
import { Badge } from './ui/badge';
import { ScrollArea } from './ui/scroll-area';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './ui/tabs';
import { Separator } from './ui/separator';
import { Textarea } from './ui/textarea';
import {
  Bell,
  AlertCircle,
  CheckCircle,
  Info,
  Settings,
  Clock,
  Eye,
  Trash2,
  MailCheck,
  Shield,
  UserCheck,
  MapPin,
  Radio,
  Send,
} from 'lucide-react';
import { toast } from 'sonner@2.0.3';
import { useTheme } from './ThemeProvider';

interface Notification {
  id: number;
  type: 'critical' | 'warning' | 'info' | 'success' | 'emergency';
  title: string;
  message: string;
  timestamp: string;
  read: boolean;
  icon: any;
  action: string;
  incidentId?: string;
  location?: string;
  confidence?: number;
  requiresAck?: boolean;
  ackBy?: string;
  ackTime?: string;
  fieldResponse?: string;
}

interface FieldResponse {
  incidentId: string;
  status: 'dispatched' | 'on_scene' | 'resolved' | 'false_alarm';
  notes: string;
  timestamp: string;
}

export function NotificationCenter() {
  const [filter, setFilter] = useState('all');
  const [activeTab, setActiveTab] = useState('alerts');
  const [fieldResponse, setFieldResponse] = useState('');
  const [notifications, setNotifications] = useState<Notification[]>([]);

  // Real-time incident WebSocket integration
  const handleIncident = useCallback((incident: any) => {
    // Prevent duplicate notifications by incidentId
    setNotifications(prev => {
      if (prev.some(n => n.incidentId === incident.id)) return prev;
      // Map backend incident to Notification type
      let type: Notification['type'] = 'info';
      let icon = Info;
      let title = 'Incident detected';
      let message = incident.description || '';
      if (incident.type === 'violence') {
        type = 'emergency';
        icon = AlertCircle;
        title = `EMERGENCY: Violence Detected`;
        message = `Violence detected at ${incident.location || incident.cameraId}`;
      } else if (incident.type === 'crash') {
        type = 'emergency';
        icon = AlertCircle;
        title = `EMERGENCY: Car Crash Detected`;
        message = `Crash detected at ${incident.location || incident.cameraId}`;
      }
      return [
        {
          id: Date.now(),
          type,
          title,
          message,
          timestamp: new Date().toLocaleTimeString(),
          read: false,
          icon,
          action: 'View Incident',
          incidentId: incident.id,
          location: incident.location,
          confidence: incident.confidence,
          requiresAck: true,
        },
        ...prev,
      ];
    });
  }, []);
  useIncidentWebSocket(handleIncident);
  const { theme } = useTheme();

  const [fieldResponses, setFieldResponses] = useState<FieldResponse[]>([
    {
      incidentId: 'INC-1234',
      status: 'resolved',
      notes: 'Situation contained, no injuries reported. Area secured.',
      timestamp: '1 hour ago',
    },
    {
      incidentId: 'INC-1228',
      status: 'on_scene',
      notes: 'Unit dispatched, assessing situation. Traffic control established.',
      timestamp: '45 minutes ago',
    },
  ]);

  const filteredNotifications = notifications.filter((notif) => {
    if (filter === 'unread') return !notif.read;
    if (filter === 'emergency') return notif.type === 'emergency';
    if (filter === 'critical') return notif.type === 'critical';
    if (filter === 'warning') return notif.type === 'warning';
    return true;
  });

  const unreadCount = notifications.filter((n) => !n.read).length;
  const emergencyAlerts = notifications.filter((n) => n.type === 'emergency' && !n.read).length;
  const pendingAck = notifications.filter((n) => n.requiresAck && !n.ackBy).length;

  const handleMarkAsRead = (id: number) => {
    setNotifications(prev => 
      prev.map(notif => 
        notif.id === id ? { ...notif, read: true } : notif
      )
    );
    toast.success('Notification marked as read');
  };

  const handleAcknowledgeEmergency = (id: number) => {
    setNotifications(prev =>
      prev.map(notif =>
        notif.id === id
          ? {
              ...notif,
              requiresAck: false,
              ackBy: 'Security Authority',
              ackTime: 'Just now',
              read: true,
            }
          : notif
      )
    );
    toast.success('Emergency alert acknowledged');
  };

  const handleSendFieldResponse = (incidentId: string) => {
    if (!fieldResponse.trim()) {
      toast.error('Please enter field response details');
      return;
    }

    const newResponse: FieldResponse = {
      incidentId,
      status: 'on_scene',
      notes: fieldResponse,
      timestamp: 'Just now',
    };

    setFieldResponses(prev => [newResponse, ...prev]);
    setFieldResponse('');

    // Update notification with response
    setNotifications(prev =>
      prev.map(notif =>
        notif.incidentId === incidentId
          ? {
              ...notif,
              fieldResponse: fieldResponse,
              requiresAck: false,
              ackBy: 'Field Unit',
              ackTime: 'Just now',
            }
          : notif
      )
    );

    toast.success('Field response report sent');
  };

  const handleDelete = (id: number) => {
    setNotifications(prev => prev.filter(notif => notif.id !== id));
    toast.success('Notification deleted');
  };

  const handleMarkAllAsRead = () => {
    setNotifications(prev => prev.map(notif => ({ ...notif, read: true })));
    toast.success('All notifications marked as read');
  };

  const getNotificationStyle = (type: string) => {
    switch (type) {
      case 'emergency':
        return {
          bg: 'bg-red-100 dark:bg-red-500/10',
          border: 'border-red-200 dark:border-red-500/30',
          icon: 'text-red-600 dark:text-red-400',
          badge: 'bg-red-100 text-red-800 border-red-200 dark:bg-red-500/20 dark:text-red-400 dark:border-red-500/30 animate-pulse',
        };
      case 'critical':
        return {
          bg: 'bg-red-50 dark:bg-red-500/5',
          border: 'border-red-100 dark:border-red-500/20',
          icon: 'text-red-600 dark:text-red-400',
          badge: 'bg-red-100 text-red-800 border-red-200 dark:bg-red-500/20 dark:text-red-400 dark:border-red-500/30',
        };
      case 'warning':
        return {
          bg: 'bg-amber-100 dark:bg-amber-500/10',
          border: 'border-amber-200 dark:border-amber-500/30',
          icon: 'text-amber-600 dark:text-amber-400',
          badge: 'bg-amber-100 text-amber-800 border-amber-200 dark:bg-amber-500/20 dark:text-amber-400 dark:border-amber-500/30',
        };
      case 'success':
        return {
          bg: 'bg-green-100 dark:bg-green-500/10',
          border: 'border-green-200 dark:border-green-500/30',
          icon: 'text-green-600 dark:text-green-400',
          badge: 'bg-green-100 text-green-800 border-green-200 dark:bg-green-500/20 dark:text-green-400 dark:border-green-500/30',
        };
      default:
        return {
          bg: 'bg-blue-100 dark:bg-blue-500/10',
          border: 'border-blue-200 dark:border-blue-500/30',
          icon: 'text-blue-600 dark:text-blue-400',
          badge: 'bg-blue-100 text-blue-800 border-blue-200 dark:bg-blue-500/20 dark:text-blue-400 dark:border-blue-500/30',
        };
    }
  };

  const getStatusBadge = (status: string) => {
    switch (status) {
      case 'dispatched':
        return 'bg-blue-100 text-blue-800 border-blue-200 dark:bg-blue-500/20 dark:text-blue-400 dark:border-blue-500/30';
      case 'on_scene':
        return 'bg-amber-100 text-amber-800 border-amber-200 dark:bg-amber-500/20 dark:text-amber-400 dark:border-amber-500/30';
      case 'resolved':
        return 'bg-green-100 text-green-800 border-green-200 dark:bg-green-500/20 dark:text-green-400 dark:border-green-500/30';
      case 'false_alarm':
        return 'bg-gray-100 text-gray-800 border-gray-200 dark:bg-gray-500/20 dark:text-gray-400 dark:border-gray-500/30';
      default:
        return 'bg-gray-100 text-gray-800 border-gray-200 dark:bg-gray-500/20 dark:text-gray-400 dark:border-gray-500/30';
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
        <div>
          <h2 className="text-2xl font-bold text-foreground mb-1 flex items-center gap-2">
            Notification Center
            {unreadCount > 0 && (
              <Badge variant="destructive" className="ml-2">
                {unreadCount} unread
              </Badge>
            )}
          </h2>
          <p className="text-muted-foreground">Emergency alerts, acknowledgements, and field responses</p>
        </div>
        <div className="flex gap-2">
          <Button
            variant="outline"
            className="border-border text-foreground hover:bg-muted"
            onClick={handleMarkAllAsRead}
          >
            <MailCheck className="w-4 h-4 mr-2" />
            Mark All Read
          </Button>
          <Button variant="ghost" className="text-muted-foreground hover:text-foreground">
            <Settings className="w-4 h-4" />
          </Button>
        </div>
      </div>

      {/* Emergency Alert Summary */}
      {emergencyAlerts > 0 && (
        <Card className="bg-red-50 dark:bg-red-500/10 border-red-200 dark:border-red-500/30">
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <AlertCircle className="w-6 h-6 text-red-600 dark:text-red-400" />
                <div>
                  <h3 className="font-semibold text-red-800 dark:text-red-300">
                    {emergencyAlerts} Emergency Alert{emergencyAlerts > 1 ? 's' : ''} Requiring Attention
                  </h3>
                  <p className="text-sm text-red-700 dark:text-red-400">
                    Immediate acknowledgement and response required
                  </p>
                </div>
              </div>
              <Badge variant="destructive" className="animate-pulse">
                <Shield className="w-3 h-3 mr-1" />
                EMERGENCY
              </Badge>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Tabs */}
      <Tabs value={activeTab} onValueChange={setActiveTab}>
        <TabsList className="grid w-full grid-cols-3 bg-muted">
          <TabsTrigger value="alerts" className="data-[state=active]:bg-blue-600 data-[state=active]:text-white">
            <Bell className="w-4 h-4 mr-2" />
            Alerts ({notifications.length})
          </TabsTrigger>
          <TabsTrigger value="emergency" className="data-[state=active]:bg-red-600 data-[state=active]:text-white">
            <Shield className="w-4 h-4 mr-2" />
            Emergency ({emergencyAlerts})
          </TabsTrigger>
          <TabsTrigger value="responses" className="data-[state=active]:bg-blue-600 data-[state=active]:text-white">
            <UserCheck className="w-4 h-4 mr-2" />
            Field Responses
          </TabsTrigger>
        </TabsList>

        {/* Alerts Tab */}
        <TabsContent value="alerts" className="space-y-6">
          {/* Filter Tabs */}
          <Card className="bg-card/80 backdrop-blur border-border">
            <CardContent className="p-4">
              <Tabs value={filter} onValueChange={setFilter}>
                <TabsList className="grid w-full grid-cols-5 bg-muted">
                  <TabsTrigger value="all" className="data-[state=active]:bg-blue-600 data-[state=active]:text-white">
                    All
                  </TabsTrigger>
                  <TabsTrigger value="unread" className="data-[state=active]:bg-blue-600 data-[state=active]:text-white">
                    Unread ({unreadCount})
                  </TabsTrigger>
                  <TabsTrigger value="emergency" className="data-[state=active]:bg-red-600 data-[state=active]:text-white">
                    Emergency
                  </TabsTrigger>
                  <TabsTrigger value="critical" className="data-[state=active]:bg-blue-600 data-[state=active]:text-white">
                    Critical
                  </TabsTrigger>
                  <TabsTrigger value="warning" className="data-[state=active]:bg-blue-600 data-[state=active]:text-white">
                    Warnings
                  </TabsTrigger>
                </TabsList>
              </Tabs>
            </CardContent>
          </Card>

          {/* Notifications List */}
          <Card className="bg-card/80 backdrop-blur border-border">
            <CardContent className="p-0">
              <ScrollArea className="h-[600px]">
                <div className="p-4 space-y-3">
                  {filteredNotifications.map((notification, index) => {
                    const Icon = notification.icon;
                    const styles = getNotificationStyle(notification.type);

                    return (
                      <div key={notification.id}>
                        <div
                          className={`p-4 rounded-lg border ${styles.border} ${styles.bg} ${
                            !notification.read ? 'ring-1 ring-blue-500/20' : ''
                          } transition-all hover:bg-opacity-80`}
                        >
                          <div className="flex items-start gap-4">
                            <div
                              className={`w-10 h-10 rounded-lg flex items-center justify-center flex-shrink-0 ${styles.bg} border ${styles.border}`}
                            >
                              <Icon className={`w-5 h-5 ${styles.icon}`} />
                            </div>

                            <div className="flex-1 min-w-0">
                              <div className="flex items-start justify-between gap-2 mb-1">
                                <div className="flex items-center gap-2">
                                  <h3 className="text-foreground">{notification.title}</h3>
                                  {!notification.read && (
                                    <div className="w-2 h-2 rounded-full bg-blue-600" />
                                  )}
                                </div>
                                <div className="flex items-center gap-2">
                                  {notification.confidence && (
                                    <Badge className="bg-green-100 text-green-800 border-green-200 dark:bg-green-500/20 dark:text-green-400 dark:border-green-500/30">
                                      {notification.confidence}% confidence
                                    </Badge>
                                  )}
                                  <Badge className={styles.badge} variant="outline">
                                    {notification.type}
                                  </Badge>
                                </div>
                              </div>

                              <p className="text-sm text-muted-foreground mb-2">{notification.message}</p>

                              {/* Incident Details */}
                              {(notification.location || notification.incidentId) && (
                                <div className="flex items-center gap-4 text-xs text-muted-foreground mb-2">
                                  {notification.incidentId && (
                                    <span>ID: {notification.incidentId}</span>
                                  )}
                                  {notification.location && (
                                    <span className="flex items-center gap-1">
                                      <MapPin className="w-3 h-3" />
                                      {notification.location}
                                    </span>
                                  )}
                                </div>
                              )}

                              {/* Acknowledgement Status */}
                              {notification.ackBy && (
                                <div className="flex items-center gap-2 text-xs text-green-600 dark:text-green-400 mb-2">
                                  <UserCheck className="w-3 h-3" />
                                  Acknowledged by {notification.ackBy} • {notification.ackTime}
                                </div>
                              )}

                              {/* Field Response */}
                              {notification.fieldResponse && (
                                <div className="bg-green-50 dark:bg-green-500/10 border border-green-200 dark:border-green-500/20 rounded p-2 mb-2">
                                  <p className="text-xs text-green-800 dark:text-green-300">
                                    <strong>Field Response:</strong> {notification.fieldResponse}
                                  </p>
                                </div>
                              )}

                              <div className="flex items-center justify-between">
                                <div className="flex items-center gap-2 text-xs text-muted-foreground">
                                  <Clock className="w-3 h-3" />
                                  {notification.timestamp}
                                </div>

                                <div className="flex items-center gap-2">
                                  {/* View button navigates to incident video if available */}
                                  {notification.incidentId && (
                                    <Button
                                      size="sm"
                                      className="h-7 bg-blue-600 hover:bg-blue-700 text-white"
                                      onClick={() => {
                                        // Custom event: navigate to incident modal
                                        window.dispatchEvent(new CustomEvent('view-incident', { detail: { incidentId: notification.incidentId } }));
                                      }}
                                    >
                                      <Eye className="w-4 h-4 mr-1" />
                                      View
                                    </Button>
                                  )}
                                  {/* Emergency Acknowledgement */}
                                  {notification.requiresAck && (
                                    <Button
                                      size="sm"
                                      className="h-7 bg-red-600 hover:bg-red-700 text-white"
                                      onClick={() => handleAcknowledgeEmergency(notification.id)}
                                    >
                                      <Shield className="w-3 h-3 mr-1" />
                                      Acknowledge
                                    </Button>
                                  )}
                                  {!notification.read && !notification.requiresAck && (
                                    <Button
                                      size="icon"
                                      variant="ghost"
                                      className="h-7 w-7 text-muted-foreground hover:text-foreground"
                                      onClick={() => handleMarkAsRead(notification.id)}
                                    >
                                      <MailCheck className="w-4 h-4" />
                                    </Button>
                                  )}
                                  {/* Exit/X button for notification */}
                                  <Button
                                    size="icon"
                                    variant="ghost"
                                    className="h-7 w-7 text-muted-foreground hover:text-red-600 dark:hover:text-red-400"
                                    onClick={() => handleDelete(notification.id)}
                                  >
                                    <X className="w-4 h-4" />
                                  </Button>
                                </div>
                              </div>
                            </div>
                          </div>
                        </div>
                        {index < filteredNotifications.length - 1 && (
                          <Separator className="my-3 bg-border" />
                        )}
                      </div>
                    );
                  })}
                </div>
              </ScrollArea>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Emergency Tab */}
        <TabsContent value="emergency">
          <Card className="bg-card/80 backdrop-blur border-border">
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-card-foreground">
                <Shield className="w-5 h-5 text-red-600 dark:text-red-400" />
                Emergency Alerts Requiring Acknowledgement ({pendingAck})
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {notifications
                  .filter(notif => notif.type === 'emergency' && notif.requiresAck)
                  .map((notification) => (
                    <div
                      key={notification.id}
                      className="p-4 rounded-lg border border-red-200 dark:border-red-500/30 bg-red-50 dark:bg-red-500/10"
                    >
                      <div className="flex items-start justify-between mb-3">
                        <div>
                          <h3 className="font-semibold text-red-800 dark:text-red-300 text-lg">
                            {notification.title}
                          </h3>
                          <p className="text-red-700 dark:text-red-400 mt-1">
                            {notification.message}
                          </p>
                        </div>
                        <Badge variant="destructive" className="animate-pulse">
                          EMERGENCY
                        </Badge>
                      </div>

                      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
                        <div className="space-y-2">
                          <div className="flex items-center gap-2 text-sm">
                            <MapPin className="w-4 h-4 text-red-600" />
                            <span className="text-foreground">Location:</span>
                            <span>{notification.location}</span>
                          </div>
                          <div className="flex items-center gap-2 text-sm">
                            <Radio className="w-4 h-4 text-red-600" />
                            <span className="text-foreground">Confidence:</span>
                            <Badge className="bg-green-100 text-green-800">
                              {notification.confidence}%
                            </Badge>
                          </div>
                        </div>
                        <div className="space-y-2">
                          <div className="flex items-center gap-2 text-sm">
                            <Clock className="w-4 h-4 text-red-600" />
                            <span className="text-foreground">Detected:</span>
                            <span>{notification.timestamp}</span>
                          </div>
                          <div className="flex items-center gap-2 text-sm">
                            <Eye className="w-4 h-4 text-red-600" />
                            <span className="text-foreground">Incident ID:</span>
                            <span>{notification.incidentId}</span>
                          </div>
                        </div>
                      </div>

                      {/* Field Response Input */}
                      <div className="space-y-3">
                        <Textarea
                          placeholder="Enter field response details (situation assessment, actions taken, resources deployed)..."
                          value={fieldResponse}
                          onChange={(e) => setFieldResponse(e.target.value)}
                          className="min-h-[80px] border-border"
                        />
                        <div className="flex gap-2">
                          <Button
                            className="bg-red-600 hover:bg-red-700 text-white"
                            onClick={() => handleAcknowledgeEmergency(notification.id)}
                          >
                            <Shield className="w-4 h-4 mr-2" />
                            Acknowledge Alert
                          </Button>
                          <Button
                            className="bg-blue-600 hover:bg-blue-700 text-white"
                            onClick={() => notification.incidentId && handleSendFieldResponse(notification.incidentId)}
                          >
                            <Send className="w-4 h-4 mr-2" />
                            Send Field Response
                          </Button>
                        </div>
                      </div>
                    </div>
                  ))}
                
                {pendingAck === 0 && (
                  <div className="text-center py-8 text-muted-foreground">
                    <CheckCircle className="w-12 h-12 mx-auto mb-4 text-green-500" />
                    <p className="text-lg font-medium">All emergency alerts have been acknowledged</p>
                    <p className="text-sm">No pending emergency alerts requiring attention</p>
                  </div>
                )}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Field Responses Tab */}
        <TabsContent value="responses">
          <Card className="bg-card/80 backdrop-blur border-border">
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-card-foreground">
                <UserCheck className="w-5 h-5 text-blue-600 dark:text-blue-400" />
                Field Response Reports ({fieldResponses.length})
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {fieldResponses.map((response, index) => (
                  <div
                    key={index}
                    className="p-4 rounded-lg border border-border bg-muted/50"
                  >
                    <div className="flex items-start justify-between mb-3">
                      <div>
                        <h3 className="font-semibold text-foreground">
                          Incident {response.incidentId}
                        </h3>
                        <Badge className={getStatusBadge(response.status)}>
                          {response.status.replace('_', ' ').toUpperCase()}
                        </Badge>
                      </div>
                      <div className="text-sm text-muted-foreground">
                        {response.timestamp}
                      </div>
                    </div>
                    <p className="text-sm text-foreground">{response.notes}</p>
                  </div>
                ))}
                
                {fieldResponses.length === 0 && (
                  <div className="text-center py-8 text-muted-foreground">
                    <UserCheck className="w-12 h-12 mx-auto mb-4 text-blue-500" />
                    <p className="text-lg font-medium">No field responses yet</p>
                    <p className="text-sm">Field response reports will appear here when submitted</p>
                  </div>
                )}
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>

      {/* Notification Settings */}
      <Card className="bg-card/80 backdrop-blur border-border">
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-card-foreground">
            <Settings className="w-5 h-5 text-blue-600 dark:text-blue-400" />
            Notification Preferences
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex items-center justify-between p-3 rounded-lg bg-muted/50 border border-border">
            <div>
              <p className="text-foreground mb-1">Emergency Alerts</p>
              <p className="text-sm text-muted-foreground">
                Immediate notifications for high-confidence incidents requiring Security Authority response
              </p>
            </div>
            <Badge className="bg-green-100 text-green-800 border-green-200 dark:bg-green-500/20 dark:text-green-400 dark:border-green-500/30">
              Enabled
            </Badge>
          </div>

          <div className="flex items-center justify-between p-3 rounded-lg bg-muted/50 border border-border">
            <div>
              <p className="text-foreground mb-1">Critical Incident Alerts</p>
              <p className="text-sm text-muted-foreground">
                High-priority notifications for confirmed incidents
              </p>
            </div>
            <Badge className="bg-green-100 text-green-800 border-green-200 dark:bg-green-500/20 dark:text-green-400 dark:border-green-500/30">
              Enabled
            </Badge>
          </div>

          <div className="flex items-center justify-between p-3 rounded-lg bg-muted/50 border border-border">
            <div>
              <p className="text-foreground mb-1">Field Response Updates</p>
              <p className="text-sm text-muted-foreground">
                Notifications when field units submit response reports
              </p>
            </div>
            <Badge className="bg-green-100 text-green-800 border-green-200 dark:bg-green-500/20 dark:text-green-400 dark:border-green-500/30">
              Enabled
            </Badge>
          </div>

          <div className="flex items-center justify-between p-3 rounded-lg bg-muted/50 border border-border">
            <div>
              <p className="text-foreground mb-1">System Health Reports</p>
              <p className="text-sm text-muted-foreground">Daily system performance summaries</p>
            </div>
            <Badge className="bg-gray-100 text-gray-800 border-gray-200 dark:bg-gray-500/20 dark:text-gray-400 dark:border-gray-500/30">
              Disabled
            </Badge>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}