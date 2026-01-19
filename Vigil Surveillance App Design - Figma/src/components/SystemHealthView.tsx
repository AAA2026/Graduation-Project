import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';
import { Badge } from './ui/badge';
import { Progress } from './ui/progress';
import { vigilClasses } from './vigil-theme';
import {
  Activity,
  Database,
  Cpu,
  HardDrive,
  Wifi,
  Server,
  AlertTriangle,
  CheckCircle,
  TrendingUp,
  Clock,
} from 'lucide-react';

interface SystemMetric {
  name: string;
  value: number;
  unit: string;
  status: 'healthy' | 'warning' | 'critical';
  icon: React.ComponentType<{ className?: string }>;
}

export function SystemHealthView() {
  const systemMetrics: SystemMetric[] = [
    { name: 'CPU Usage', value: 45, unit: '%', status: 'healthy', icon: Cpu },
    { name: 'Memory Usage', value: 68, unit: '%', status: 'warning', icon: HardDrive },
    { name: 'Database Load', value: 32, unit: '%', status: 'healthy', icon: Database },
    { name: 'API Latency', value: 145, unit: 'ms', status: 'healthy', icon: Activity },
    { name: 'Storage Used', value: 73, unit: '%', status: 'warning', icon: Server },
    { name: 'Network Traffic', value: 28, unit: 'Mbps', status: 'healthy', icon: Wifi },
  ];

  const services = [
    { name: 'API Gateway', status: 'online', uptime: '99.98%', responseTime: '45ms' },
    { name: 'AI Detection Engine', status: 'online', uptime: '99.95%', responseTime: '120ms' },
    { name: 'Frame Processor', status: 'online', uptime: '99.97%', responseTime: '89ms' },
    { name: 'Notification Service', status: 'online', uptime: '100%', responseTime: '12ms' },
    { name: 'Incident Management', status: 'online', uptime: '99.99%', responseTime: '34ms' },
    { name: 'Report & Analytics', status: 'online', uptime: '99.94%', responseTime: '78ms' },
    { name: 'User & Auth Service', status: 'online', uptime: '100%', responseTime: '23ms' },
    { name: 'Model Admin Service', status: 'online', uptime: '99.92%', responseTime: '156ms' },
  ];

  const aiModels = [
    { name: 'Violence Detection', version: 'v2.4.1', accuracy: 94.2, status: 'active', lastUpdated: '2 days ago' },
    { name: 'Car Crash Detection', version: 'v1.8.3', accuracy: 91.7, status: 'active', lastUpdated: '5 days ago' },
    { name: 'Assault Detection', version: 'v3.1.0', accuracy: 89.5, status: 'active', lastUpdated: '1 week ago' },
    { name: 'Theft Detection', version: 'v1.5.2', accuracy: 87.3, status: 'training', lastUpdated: '3 days ago' },
  ];

  const recentEvents = [
    { time: '2 minutes ago', event: 'AI model retrained successfully', type: 'success' },
    { time: '15 minutes ago', event: 'Database backup completed', type: 'info' },
    { time: '1 hour ago', event: 'High memory usage detected', type: 'warning' },
    { time: '3 hours ago', event: 'System health check passed', type: 'success' },
  ];

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'healthy':
      case 'online':
      case 'active':
        return vigilClasses.badgeSuccess;
      case 'warning':
        return vigilClasses.badgeWarning;
      case 'critical':
      case 'offline':
        return vigilClasses.badgeDanger;
      case 'training':
        return vigilClasses.badgeInfo;
      default:
        return vigilClasses.badgeInfo;
    }
  };

  const getProgressColor = (status: string) => {
    switch (status) {
      case 'healthy':
        return 'bg-green-500';
      case 'warning':
        return 'bg-amber-500';
      case 'critical':
        return 'bg-red-500';
      default:
        return 'bg-blue-500 dark:bg-cyan-500';
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center gap-2">
        <Activity className="w-7 h-7 text-blue-600 dark:text-blue-400" />
        <h1 className="text-2xl font-bold tracking-tight text-blue-900 dark:text-blue-400">System Health</h1>
      </div>
      <p className="text-base text-muted-foreground">Monitor system performance, service uptime, and AI model status in real time.</p>

      {/* System Metrics Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {systemMetrics.map((metric) => {
          const Icon = metric.icon;
          return (
            <Card key={metric.name} className={vigilClasses.card}>
              <CardContent className="p-6">
                <div className="flex items-start justify-between mb-4">
                  <div className="flex items-center gap-3">
                    <div className={`p-2 rounded-lg ${
                      metric.status === 'healthy' 
                        ? 'bg-green-100 dark:bg-green-900/30' 
                        : metric.status === 'warning'
                        ? 'bg-amber-100 dark:bg-amber-900/30'
                        : 'bg-red-100 dark:bg-red-900/30'
                    }`}>
                      <Icon className={`w-5 h-5 ${
                        metric.status === 'healthy' 
                          ? 'text-green-600 dark:text-green-400' 
                          : metric.status === 'warning'
                          ? 'text-amber-600 dark:text-amber-400'
                          : 'text-red-600 dark:text-red-400'
                      }`} />
                    </div>
                    <div>
                      <p className={`text-sm ${vigilClasses.textSecondary}`}>{metric.name}</p>
                      <div className="flex items-baseline gap-1 mt-1">
                        <span className={`text-2xl ${vigilClasses.textPrimary}`}>{metric.value}</span>
                        <span className={`text-sm ${vigilClasses.textMuted}`}>{metric.unit}</span>
                      </div>
                    </div>
                  </div>
                  <Badge className={getStatusColor(metric.status)}>
                    {metric.status}
                  </Badge>
                </div>
                <Progress value={metric.value} className="h-2" indicatorClassName={getProgressColor(metric.status)} />
              </CardContent>
            </Card>
          );
        })}
      </div>

      {/* Services Status */}
      <Card className={vigilClasses.card}>
        <CardHeader>
          <CardTitle className={vigilClasses.textPrimary}>Backend Services</CardTitle>
          <CardDescription>Real-time status of all microservices</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {services.map((service) => (
              <div
                key={service.name}
                className={`p-4 rounded-lg border ${vigilClasses.border} hover:bg-gray-50 dark:hover:bg-gray-800/50 transition-colors`}
              >
                <div className="flex items-center justify-between mb-2">
                  <div className="flex items-center gap-2">
                    <CheckCircle className="w-4 h-4 text-green-600 dark:text-green-400" />
                    <span className={vigilClasses.textPrimary}>{service.name}</span>
                  </div>
                  <Badge className={getStatusColor(service.status)}>{service.status}</Badge>
                </div>
                <div className="grid grid-cols-2 gap-4 mt-3">
                  <div>
                    <p className={`text-xs ${vigilClasses.textMuted}`}>Uptime</p>
                    <p className={`text-sm ${vigilClasses.textSecondary}`}>{service.uptime}</p>
                  </div>
                  <div>
                    <p className={`text-xs ${vigilClasses.textMuted}`}>Response Time</p>
                    <p className={`text-sm ${vigilClasses.textSecondary}`}>{service.responseTime}</p>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* AI Models Status */}
      <Card className={vigilClasses.card}>
        <CardHeader>
          <CardTitle className={vigilClasses.textPrimary}>AI Detection Models</CardTitle>
          <CardDescription>Current model versions and performance metrics</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {aiModels.map((model) => (
              <div
                key={model.name}
                className={`p-4 rounded-lg border ${vigilClasses.border}`}
              >
                <div className="flex items-center justify-between mb-3">
                  <div>
                    <h4 className={vigilClasses.textPrimary}>{model.name}</h4>
                    <p className={`text-sm ${vigilClasses.textMuted}`}>Version {model.version}</p>
                  </div>
                  <Badge className={getStatusColor(model.status)}>{model.status}</Badge>
                </div>
                <div className="grid grid-cols-3 gap-4">
                  <div>
                    <p className={`text-xs ${vigilClasses.textMuted} mb-1`}>Accuracy</p>
                    <div className="flex items-center gap-2">
                      <TrendingUp className="w-4 h-4 text-green-600 dark:text-green-400" />
                      <span className={vigilClasses.textPrimary}>{model.accuracy}%</span>
                    </div>
                  </div>
                  <div>
                    <p className={`text-xs ${vigilClasses.textMuted} mb-1`}>Status</p>
                    <span className={vigilClasses.textSecondary}>{model.status === 'active' ? 'Running' : 'Training'}</span>
                  </div>
                  <div>
                    <p className={`text-xs ${vigilClasses.textMuted} mb-1`}>Last Updated</p>
                    <div className="flex items-center gap-1">
                      <Clock className={`w-3 h-3 ${vigilClasses.textMuted}`} />
                      <span className={`text-sm ${vigilClasses.textSecondary}`}>{model.lastUpdated}</span>
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Recent System Events */}
      <Card className={vigilClasses.card}>
        <CardHeader>
          <CardTitle className={vigilClasses.textPrimary}>Recent System Events</CardTitle>
          <CardDescription>Latest system activities and alerts</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-3">
            {recentEvents.map((event, index) => (
              <div
                key={index}
                className={`flex items-start gap-3 p-3 rounded-lg hover:bg-gray-50 dark:hover:bg-gray-800/50 transition-colors`}
              >
                <div className="mt-0.5">
                  {event.type === 'success' && <CheckCircle className="w-5 h-5 text-green-600 dark:text-green-400" />}
                  {event.type === 'warning' && <AlertTriangle className="w-5 h-5 text-amber-600 dark:text-amber-400" />}
                  {event.type === 'info' && <Activity className="w-5 h-5 text-blue-600 dark:text-cyan-400" />}
                </div>
                <div className="flex-1">
                  <p className={vigilClasses.textPrimary}>{event.event}</p>
                  <p className={`text-sm ${vigilClasses.textMuted}`}>{event.time}</p>
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  );
}