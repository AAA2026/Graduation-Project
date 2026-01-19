import { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Badge } from './ui/badge';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './ui/select';
import { Button } from './ui/button';
import {
  LineChart,
  Line,
  BarChart,
  Bar,
  PieChart,
  Pie,
  Cell,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';
import {
  BarChart3,
  TrendingUp,
  TrendingDown,
  Clock,
  AlertCircle,
  CheckCircle,
  Activity,
  Calendar,
  Filter
} from 'lucide-react';
import { useTheme } from './ThemeProvider';
import { vigilColors, getVigilColor, getChartColors } from './vigil-theme';
import { toast } from 'sonner';

export function AnalyticsView() {
  const { theme } = useTheme();
  // For type safety
  const resolvedTheme = theme === 'dark' ? 'dark' : 'light';
  const chartPalette = getChartColors(resolvedTheme);

  const [timeRange, setTimeRange] = useState('7days');
  const [isLoading, setIsLoading] = useState(false);

  // Initial data
  const [data, setData] = useState({
    incidentsPerDay: [
      { date: 'Mon', violence: 12, crash: 8, total: 20 },
      { date: 'Tue', violence: 15, crash: 11, total: 26 },
      { date: 'Wed', violence: 9, crash: 6, total: 15 },
      { date: 'Thu', violence: 18, crash: 14, total: 32 },
      { date: 'Fri', violence: 22, crash: 17, total: 39 },
      { date: 'Sat', violence: 28, crash: 21, total: 49 },
      { date: 'Sun', violence: 19, crash: 13, total: 32 },
    ],
    pills: {
      aiAccuracy: 98.7,
      resolutionRate: 71,
    }
  });

  // Effect to simulation data loading when time range changes
  useEffect(() => {
    setIsLoading(true);
    // Simulate API delay
    const timer = setTimeout(() => {
      if (timeRange === '7days') {
        setData({
          incidentsPerDay: [
            { date: 'Mon', violence: 12, crash: 8, total: 20 },
            { date: 'Tue', violence: 15, crash: 11, total: 26 },
            { date: 'Wed', violence: 9, crash: 6, total: 15 },
            { date: 'Thu', violence: 18, crash: 14, total: 32 },
            { date: 'Fri', violence: 22, crash: 17, total: 39 },
            { date: 'Sat', violence: 28, crash: 21, total: 49 },
            { date: 'Sun', violence: 19, crash: 13, total: 32 },
          ],
          pills: { aiAccuracy: 98.7, resolutionRate: 71 }
        });
      } else if (timeRange === '30days') {
        setData({
          incidentsPerDay: [
            { date: 'Week 1', violence: 45, crash: 32, total: 77 },
            { date: 'Week 2', violence: 52, crash: 28, total: 80 },
            { date: 'Week 3', violence: 48, crash: 35, total: 83 },
            { date: 'Week 4', violence: 61, crash: 42, total: 103 },
          ],
          pills: { aiAccuracy: 97.4, resolutionRate: 75 }
        });
      } else if (timeRange === '90days') {
        setData({
          incidentsPerDay: [
            { date: 'Month 1', violence: 180, crash: 120, total: 300 },
            { date: 'Month 2', violence: 210, crash: 145, total: 355 },
            { date: 'Month 3', violence: 195, crash: 130, total: 325 },
          ],
          pills: { aiAccuracy: 96.8, resolutionRate: 82 }
        });
      }
      setIsLoading(false);
      toast.success(`Analytics updated`);
    }, 600);
    return () => clearTimeout(timer);
  }, [timeRange]);

  const responseTimeData = [
    { hour: '00:00', avgTime: 2.8 },
    { hour: '04:00', avgTime: 2.1 },
    { hour: '08:00', avgTime: 3.2 },
    { hour: '12:00', avgTime: 2.9 },
    { hour: '16:00', avgTime: 3.5 },
    { hour: '20:00', avgTime: 3.1 },
  ];

  const incidentsByZone = [
    { name: 'Downtown', value: 142, color: '#06b6d4' },
    { name: 'Highway', value: 98, color: '#f59e0b' },
    { name: 'Residential', value: 67, color: '#10b981' },
    { name: 'Industrial', value: 45, color: '#8b5cf6' },
  ];

  const aiAccuracyTrend = [
    { week: 'Week 1', accuracy: 94.2, falsePositive: 5.8 },
    { week: 'Week 2', accuracy: 95.1, falsePositive: 4.9 },
    { week: 'Week 3', accuracy: 96.3, falsePositive: 3.7 },
    { week: 'Week 4', accuracy: 97.8, falsePositive: 2.2 },
    { week: 'Week 5', accuracy: 98.7, falsePositive: 1.3 },
  ];

  const alertResolutionData = [
    { status: 'Resolved', value: 285, percentage: 71 },
    { status: 'In Progress', value: 67, percentage: 17 },
    { status: 'Pending', value: 48, percentage: 12 },
  ];

  return (
    <div className="space-y-6 text-foreground">
      {/* Header */}
      <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
        <div>
          <h2 className="text-2xl font-bold text-foreground mb-1">Analytics</h2>
          <p className="text-muted-foreground">Performance metrics and insights</p>
        </div>
        <div className="flex items-center gap-3">
          <Badge variant="outline" className="h-9 px-3 flex items-center gap-2 border-dashed">
            <Calendar className="w-3 h-3 text-muted-foreground" />
            <span>{new Date().getFullYear()}</span>
          </Badge>
          <Select value={timeRange} onValueChange={setTimeRange}>
            <SelectTrigger className="w-48 bg-card border-border text-foreground">
              <SelectValue />
            </SelectTrigger>
            <SelectContent className="bg-card border-border text-foreground">
              <SelectItem value="7days">Last 7 Days</SelectItem>
              <SelectItem value="30days">Last 30 Days</SelectItem>
              <SelectItem value="90days">Last 90 Days</SelectItem>
            </SelectContent>
          </Select>
          <Button size="icon" variant="ghost" className="h-9 w-9">
            <Filter className="w-4 h-4" />
          </Button>
        </div>
      </div>

      {/* Key Metrics */}
      <div className={`grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 transition-opacity duration-300 ${isLoading ? 'opacity-50' : 'opacity-100'}`}>
        <Card>
          <CardContent className="p-6">
            <div className="flex items-center justify-between mb-4">
              <Activity className="w-8 h-8 text-cyan-600 dark:text-cyan-400" />
              <Badge className="bg-green-100 text-green-800 border-green-200 dark:bg-green-500/20 dark:text-green-400 dark:border-green-500/30">
                <TrendingUp className="w-3 h-3 mr-1" />
                +12%
              </Badge>
            </div>
            <div className="text-3xl text-foreground mb-1">{data.pills.aiAccuracy}%</div>
            <div className="text-sm text-muted-foreground">AI Accuracy</div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-6">
            <div className="flex items-center justify-between mb-4">
              <CheckCircle className="w-8 h-8 text-green-600 dark:text-green-400" />
              <Badge className="bg-green-100 text-green-800 border-green-200 dark:bg-green-500/20 dark:text-green-400 dark:border-green-500/30">
                <TrendingUp className="w-3 h-3 mr-1" />
                +8%
              </Badge>
            </div>
            <div className="text-3xl text-foreground mb-1">{data.pills.resolutionRate}%</div>
            <div className="text-sm text-muted-foreground">Resolution Rate</div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-6">
            <div className="flex items-center justify-between mb-4">
              <Clock className="w-8 h-8 text-amber-600 dark:text-amber-400" />
              <Badge className="bg-green-100 text-green-800 border-green-200 dark:bg-green-500/20 dark:text-green-400 dark:border-green-500/30">
                <TrendingDown className="w-3 h-3 mr-1" />
                -0.3m
              </Badge>
            </div>
            <div className="text-3xl text-foreground mb-1">2.4m</div>
            <div className="text-sm text-muted-foreground">Avg Response Time</div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-6">
            <div className="flex items-center justify-between mb-4">
              <AlertCircle className="w-8 h-8 text-red-600 dark:text-red-400" />
              <Badge className="bg-amber-100 text-amber-800 border-amber-200 dark:bg-amber-500/20 dark:text-amber-400 dark:border-amber-500/30">
                <TrendingUp className="w-3 h-3 mr-1" />
                +15
              </Badge>
            </div>
            <div className="text-3xl text-foreground mb-1">213</div>
            <div className="text-sm text-muted-foreground">Total Incidents</div>
          </CardContent>
        </Card>
      </div>

      {/* Charts */}
      <div className={`grid grid-cols-1 lg:grid-cols-2 gap-6 transition-opacity duration-300 ${isLoading ? 'opacity-50' : 'opacity-100'}`}>
        {/* Incidents Per Day */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-card-foreground">
              <BarChart3 className="w-5 h-5 text-cyan-600 dark:text-cyan-400" />
              Incidents Trend
            </CardTitle>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={data.incidentsPerDay}>
                <CartesianGrid strokeDasharray="3 3" stroke={getVigilColor('border', resolvedTheme)} />
                <XAxis dataKey="date" stroke={getVigilColor('textMuted', resolvedTheme)} tick={{ fill: getVigilColor('textMuted', resolvedTheme), fontWeight: 600 }} />
                <YAxis stroke={getVigilColor('textMuted', resolvedTheme)} tick={{ fill: getVigilColor('textMuted', resolvedTheme), fontWeight: 600 }} />
                <Tooltip
                  contentStyle={{
                    backgroundColor: getVigilColor('bgCard', resolvedTheme),
                    border: `1px solid ${getVigilColor('border', resolvedTheme)}`,
                    borderRadius: '8px',
                    color: getVigilColor('textPrimary', resolvedTheme),
                  }}
                  labelStyle={{ color: getVigilColor('textPrimary', resolvedTheme), fontWeight: 700 }}
                  itemStyle={{ color: getVigilColor('textSecondary', resolvedTheme) }}
                />
                <Legend
                  wrapperStyle={{
                    color: getVigilColor('textMuted', resolvedTheme),
                    fontWeight: 600
                  }}
                />
                <Bar dataKey="violence" fill={vigilColors[resolvedTheme].danger} name="Violence" radius={[4, 4, 0, 0]} />
                <Bar dataKey="crash" fill={vigilColors[resolvedTheme].warning} name="Car Crash" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>

        {/* Response Time Trend */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-card-foreground">
              <Clock className="w-5 h-5 text-amber-600 dark:text-amber-400" />
              Response Time Trend
            </CardTitle>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={300}>
              <AreaChart data={responseTimeData}>
                <defs>
                  <linearGradient id="colorTime" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor={vigilColors[resolvedTheme].secondary} stopOpacity={0.3} />
                    <stop offset="95%" stopColor={vigilColors[resolvedTheme].secondary} stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke={getVigilColor('border', resolvedTheme)} />
                <XAxis dataKey="hour" stroke={getVigilColor('textMuted', resolvedTheme)} tick={{ fill: getVigilColor('textMuted', resolvedTheme), fontWeight: 600 }} />
                <YAxis stroke={getVigilColor('textMuted', resolvedTheme)} tick={{ fill: getVigilColor('textMuted', resolvedTheme), fontWeight: 600 }} label={{ value: 'Minutes', angle: -90, position: 'insideLeft', fill: getVigilColor('textMuted', resolvedTheme), fontWeight: 600 }} />
                <Tooltip
                  contentStyle={{
                    backgroundColor: getVigilColor('bgCard', resolvedTheme),
                    border: `1px solid ${getVigilColor('border', resolvedTheme)}`,
                    borderRadius: '8px',
                    color: getVigilColor('textPrimary', resolvedTheme),
                  }}
                  labelStyle={{ color: getVigilColor('textPrimary', resolvedTheme), fontWeight: 700 }}
                  itemStyle={{ color: getVigilColor('textSecondary', resolvedTheme) }}
                />
                <Area
                  type="monotone"
                  dataKey="avgTime"
                  stroke={vigilColors[resolvedTheme].secondary}
                  strokeWidth={2}
                  fillOpacity={1}
                  fill="url(#colorTime)"
                  name="Avg Time (min)"
                />
              </AreaChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>

        {/* Incidents by Zone */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-card-foreground">
              <AlertCircle className="w-5 h-5 text-red-600 dark:text-red-400" />
              Incidents by Zone
            </CardTitle>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={300}>
              <PieChart>
                <Pie
                  data={incidentsByZone}
                  cx="50%"
                  cy="50%"
                  innerRadius={60}
                  outerRadius={100}
                  paddingAngle={5}
                  dataKey="value"
                  label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                >
                  {incidentsByZone.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={chartPalette[index % chartPalette.length]} />
                  ))}
                </Pie>
                <Tooltip
                  contentStyle={{
                    backgroundColor: getVigilColor('bgCard', resolvedTheme),
                    border: `1px solid ${getVigilColor('border', resolvedTheme)}`,
                    borderRadius: '8px',
                    color: getVigilColor('textPrimary', resolvedTheme),
                  }}
                  labelStyle={{ color: getVigilColor('textPrimary', resolvedTheme), fontWeight: 700 }}
                  itemStyle={{ color: getVigilColor('textSecondary', resolvedTheme) }}
                />
              </PieChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>

        {/* AI Accuracy Trend */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-card-foreground">
              <Activity className="w-5 h-5 text-green-600 dark:text-green-400" />
              AI Model Performance
            </CardTitle>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={aiAccuracyTrend}>
                <CartesianGrid strokeDasharray="3 3" stroke={getVigilColor('border', resolvedTheme)} />
                <XAxis dataKey="week" stroke={getVigilColor('textMuted', resolvedTheme)} tick={{ fill: getVigilColor('textMuted', resolvedTheme), fontWeight: 600 }} />
                <YAxis stroke={getVigilColor('textMuted', resolvedTheme)} domain={[90, 100]} tick={{ fill: getVigilColor('textMuted', resolvedTheme), fontWeight: 600 }} />
                <Tooltip
                  contentStyle={{
                    backgroundColor: getVigilColor('bgCard', resolvedTheme),
                    border: `1px solid ${getVigilColor('border', resolvedTheme)}`,
                    borderRadius: '8px',
                    color: getVigilColor('textPrimary', resolvedTheme),
                  }}
                  labelStyle={{ color: getVigilColor('textPrimary', resolvedTheme), fontWeight: 700 }}
                  itemStyle={{ color: getVigilColor('textSecondary', resolvedTheme) }}
                />
                <Legend wrapperStyle={{ color: getVigilColor('textMuted', resolvedTheme), fontWeight: 600 }} />
                <Line
                  type="monotone"
                  dataKey="accuracy"
                  stroke={vigilColors[resolvedTheme].success}
                  strokeWidth={3}
                  name="Accuracy %"
                  dot={{ r: 6, fill: vigilColors[resolvedTheme].success }}
                />
                <Line
                  type="monotone"
                  dataKey="falsePositive"
                  stroke={vigilColors[resolvedTheme].danger}
                  strokeWidth={3}
                  name="False Positive %"
                  dot={{ r: 6, fill: vigilColors[resolvedTheme].danger }}
                />
              </LineChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      </div>

      {/* Alert Resolution Status */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-card-foreground">
            <CheckCircle className="w-5 h-5 text-green-600 dark:text-green-400" />
            Alert Resolution Status
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {alertResolutionData.map((item) => (
              <div key={item.status} className="space-y-2">
                <div className="flex items-center justify-between text-sm">
                  <span className="text-foreground flex items-center gap-2">
                    <div
                      className={`w-3 h-3 rounded-full ${item.status === 'Resolved'
                          ? 'bg-green-500'
                          : item.status === 'In Progress'
                            ? 'bg-amber-500'
                            : 'bg-red-500'
                        }`}
                    />
                    {item.status}
                  </span>
                  <span className="text-muted-foreground">
                    {item.value} ({item.percentage}%)
                  </span>
                </div>
                <div className="h-3 bg-muted rounded-full overflow-hidden">
                  <div
                    className={`h-full rounded-full transition-all ${item.status === 'Resolved'
                        ? 'bg-green-500'
                        : item.status === 'In Progress'
                          ? 'bg-amber-500'
                          : 'bg-red-500'
                      }`}
                    style={{ width: `${item.percentage}%` }}
                  />
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  );
}