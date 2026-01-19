import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Button } from './ui/button';
import { Badge } from './ui/badge';
import { Progress } from './ui/progress';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './ui/tabs';
import {
  Brain,
  TrendingUp,
  CheckCircle,
  XCircle,
  RefreshCw,
  Activity,
  Database,
  Zap,
  Users,
  AlertTriangle,
  Download,
  Upload,
  Swords,
  Car,
} from 'lucide-react';
import { LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, PieChart, Pie, Cell } from 'recharts';
import { toast } from 'sonner';
import { useTheme } from './ThemeProvider';
import { useState, useEffect } from 'react';

interface TrainingSample {
  id: number;
  type: 'violence' | 'crash';
  confidence: number;
  status: 'confirmed' | 'dismissed' | 'pending';
  timestamp: string;
  location: string;
  usedInTraining: boolean;
}

export function AIFeedbackSection() {
  const { theme } = useTheme();
  const [activeTab, setActiveTab] = useState('overview');
  const [retrainingProgress, setRetrainingProgress] = useState(0);
  const [isRetraining, setIsRetraining] = useState(false);
  const [feedbackData, setFeedbackData] = useState<any[]>([]);
  const API_URL = (import.meta as any).env.VITE_API_URL || '';

  // Fetch real feedback data
  useEffect(() => {
    const fetchFeedback = async () => {
      try {
        const res = await fetch(`${API_URL}/api/feedback`);
        if (res.ok) {
          const data = await res.json();
          if (Array.isArray(data)) {
            setFeedbackData(data);
          } else {
            console.error("Feedback data is not an array:", data);
            setFeedbackData([]);
          }
        }
      } catch (e) {
        console.error("Failed to fetch feedback logs:", e);
      }
    };
    fetchFeedback();
    // Poll every 10s
    const interval = setInterval(fetchFeedback, 10000);
    return () => clearInterval(interval);
  }, []);

  // Training samples from sequence diagrams
  const trainingSamples: TrainingSample[] = [
    { id: 1, type: 'violence', confidence: 94, status: 'confirmed', timestamp: '2024-11-04 14:32:15', location: 'Main St', usedInTraining: true },
    { id: 2, type: 'crash', confidence: 72, status: 'dismissed', timestamp: '2024-11-04 14:29:42', location: 'Highway 101', usedInTraining: true },
    { id: 3, type: 'violence', confidence: 65, status: 'confirmed', timestamp: '2024-11-04 14:22:08', location: 'Park Plaza', usedInTraining: true },
    { id: 4, type: 'crash', confidence: 55, status: 'dismissed', timestamp: '2024-11-04 14:16:33', location: 'Downtown Bridge', usedInTraining: false },
    { id: 5, type: 'violence', confidence: 92, status: 'confirmed', timestamp: '2024-11-04 14:08:21', location: 'Shopping District', usedInTraining: true },
    { id: 6, type: 'crash', confidence: 45, status: 'pending', timestamp: '2024-11-04 13:55:47', location: 'Riverside Parkway', usedInTraining: false },
  ];
  // People Counter Model removed


  // Model-specific accuracy trends
  const violenceAccuracyTrend = [
    { week: 'W1', accuracy: 92.1, highConfidence: 82, mediumConfidence: 15, lowConfidence: 3 },
    { week: 'W2', accuracy: 93.5, highConfidence: 84, mediumConfidence: 13, lowConfidence: 3 },
    { week: 'W3', accuracy: 94.8, highConfidence: 86, mediumConfidence: 11, lowConfidence: 3 },
    { week: 'W4', accuracy: 96.2, highConfidence: 88, mediumConfidence: 9, lowConfidence: 3 },
    { week: 'W5', accuracy: 97.3, highConfidence: 90, mediumConfidence: 7, lowConfidence: 3 },
  ];

  const crashAccuracyTrend = [
    { week: 'W1', accuracy: 95.8, highConfidence: 87, mediumConfidence: 10, lowConfidence: 3 },
    { week: 'W2', accuracy: 96.4, highConfidence: 89, mediumConfidence: 8, lowConfidence: 3 },
    { week: 'W3', accuracy: 97.1, highConfidence: 91, mediumConfidence: 7, lowConfidence: 2 },
    { week: 'W4', accuracy: 98.2, highConfidence: 93, mediumConfidence: 5, lowConfidence: 2 },
    { week: 'W5', accuracy: 98.9, highConfidence: 95, mediumConfidence: 4, lowConfidence: 1 },
  ];

  // Enhanced feedback stats by model
  const violenceFeedbackStats = [
    { category: 'Correct Predictions', value: 1247, percentage: 91, color: '#10b981' },
    { category: 'False Positives', value: 67, percentage: 5, color: '#ef4444' },
    { category: 'False Negatives', value: 28, percentage: 2, color: '#f59e0b' },
    { category: 'Pending Review', value: 25, percentage: 2, color: '#6b7280' },
  ];

  const crashFeedbackStats = [
    { category: 'Correct Predictions', value: 895, percentage: 94, color: '#10b981' },
    { category: 'False Positives', value: 32, percentage: 3, color: '#ef4444' },
    { category: 'False Negatives', value: 14, percentage: 1, color: '#f59e0b' },
    { category: 'Pending Review', value: 12, percentage: 2, color: '#6b7280' },
  ];

  // Model-specific confidence distribution
  const violenceConfidenceDistribution = [
    { name: 'High Confidence', value: 1024, color: '#10b981' },
    { name: 'Medium Confidence', value: 198, color: '#f59e0b' },
    { name: 'Low Confidence', value: 25, color: '#ef4444' },
  ];

  const crashConfidenceDistribution = [
    { name: 'High Confidence', value: 812, color: '#10b981' },
    { name: 'Medium Confidence', value: 144, color: '#f59e0b' },
    { name: 'Low Confidence', value: 18, color: '#ef4444' },
  ];

  const handleStartRetraining = async (model: 'violence' | 'crash' | 'both') => {
    setIsRetraining(true);
    setRetrainingProgress(0);

    // Simulate initial progress
    setRetrainingProgress(10);

    try {
      const res = await fetch(`${API_URL}/api/retrain`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model_type: model }) // Pass specific model if backend supports it
      });

      if (res.ok) {
        setRetrainingProgress(50);
        const data = await res.json();
        // Wait a bit to show progress
        setTimeout(() => {
          setRetrainingProgress(100);
          setIsRetraining(false);
          toast.success(`Retraining complete! ${data.message || ''}`);
        }, 1500);
      } else {
        throw new Error("Retraining failed");
      }
    } catch (e) {
      console.error(e);
      setRetrainingProgress(0);
      setIsRetraining(false);
      toast.error("Failed to start retraining task. Check backend.");
    }
  };

  const handleExportTrainingData = (model: 'violence' | 'crash' | 'both') => {
    const samplesForExport = trainingSamples.filter(sample =>
      sample.usedInTraining && (model === 'both' || sample.type === model)
    );
    toast.success(`Exported ${samplesForExport.length} training samples for ${model === 'both' ? 'both models' : model + ' model'}`);
  };

  const handleImportTrainingData = (model: 'violence' | 'crash') => {
    toast.info(`Training data import initiated for ${model} detection model`);
  };

  // Calculate metrics from sequence diagrams
  const totalSamples = trainingSamples.length;
  const violenceSamples = trainingSamples.filter(s => s.type === 'violence');
  const crashSamples = trainingSamples.filter(s => s.type === 'crash');

  const confirmedViolence = violenceSamples.filter(s => s.status === 'confirmed').length;
  const dismissedViolence = violenceSamples.filter(s => s.status === 'dismissed').length;
  const confirmedCrash = crashSamples.filter(s => s.status === 'confirmed').length;
  const dismissedCrash = crashSamples.filter(s => s.status === 'dismissed').length;

  const usedInTraining = trainingSamples.filter(s => s.usedInTraining).length;

  const getConfidenceBadgeStyle = (level: string) => {
    switch (level) {
      case 'high':
        return 'bg-green-100 text-green-800 border-green-200 dark:bg-green-500/20 dark:text-green-400 dark:border-green-500/30';
      case 'medium':
        return 'bg-amber-100 text-amber-800 border-amber-200 dark:bg-amber-500/20 dark:text-amber-400 dark:border-amber-500/30';
      case 'low':
        return 'bg-red-100 text-red-800 border-red-200 dark:bg-red-500/20 dark:text-red-400 dark:border-red-500/30';
      default:
        return 'bg-gray-100 text-gray-800 border-gray-200 dark:bg-gray-500/20 dark:text-gray-400 dark:border-gray-500/30';
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
        <div>
          <h2 className="text-2xl font-bold text-foreground mb-1">AI Feedback & Retraining</h2>
          <p className="text-muted-foreground">Monitor model performance and manage retraining datasets</p>
        </div>
        <div className="flex gap-2">
          <Button
            variant="outline"
            className="border-border text-foreground hover:bg-muted"
            onClick={() => handleExportTrainingData('both')}
          >
            <Download className="w-4 h-4 mr-2" />
            Export All Data
          </Button>
          <Button
            className="bg-blue-600 hover:bg-blue-700 text-white"
            onClick={() => handleStartRetraining('both')}
            disabled={isRetraining}
          >
            <RefreshCw className={`w-4 h-4 mr-2 ${isRetraining ? 'animate-spin' : ''}`} />
            {isRetraining ? 'Training...' : 'Retrain Both Models'}
          </Button>
        </div>
      </div>

      {/* Model Cards */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Violence Detection Model */}
        <Card className="bg-card/80 backdrop-blur border-border">
          <CardHeader className="pb-3">
            <CardTitle className="flex items-center gap-2 text-card-foreground">
              <Swords className="w-5 h-5 text-red-600 dark:text-red-400" />
              Violence Detection Model
              <Badge className="bg-green-100 text-green-800 border-green-200 dark:bg-green-500/20 dark:text-green-400 dark:border-green-500/30">
                v4.2.1
              </Badge>
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <div className="text-center">
                <div className="text-2xl font-bold text-foreground">97.3%</div>
                <div className="text-sm text-muted-foreground">Accuracy</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-foreground">{violenceSamples.length}</div>
                <div className="text-sm text-muted-foreground">Samples</div>
              </div>
            </div>

            <div className="space-y-2">
              <div className="flex justify-between text-sm">
                <span className="text-muted-foreground">Confirmed</span>
                <span className="text-foreground">{confirmedViolence}</span>
              </div>
              <div className="flex justify-between text-sm">
                <span className="text-muted-foreground">False Positives</span>
                <span className="text-foreground">{dismissedViolence}</span>
              </div>
            </div>

            <div className="flex gap-2">
              <Button
                size="sm"
                variant="outline"
                className="flex-1 border-border text-foreground hover:bg-muted"
                onClick={() => handleExportTrainingData('violence')}
              >
                <Download className="w-3 h-3 mr-1" />
                Export
              </Button>
              <Button
                size="sm"
                className="flex-1 bg-red-600 hover:bg-red-700 text-white"
                onClick={() => handleStartRetraining('violence')}
                disabled={isRetraining}
              >
                <RefreshCw className="w-3 h-3 mr-1" />
                Retrain
              </Button>
            </div>
          </CardContent>
        </Card>

        {/* Car Crash Detection Model */}
        <Card className="bg-card/80 backdrop-blur border-border">
          <CardHeader className="pb-3">
            <CardTitle className="flex items-center gap-2 text-card-foreground">
              <Car className="w-5 h-5 text-amber-600 dark:text-amber-400" />
              Car Crash Detection Model
              <Badge className="bg-green-100 text-green-800 border-green-200 dark:bg-green-500/20 dark:text-green-400 dark:border-green-500/30">
                v3.8.2
              </Badge>
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <div className="text-center">
                <div className="text-2xl font-bold text-foreground">98.9%</div>
                <div className="text-sm text-muted-foreground">Accuracy</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-foreground">{crashSamples.length}</div>
                <div className="text-sm text-muted-foreground">Samples</div>
              </div>
            </div>

            <div className="space-y-2">
              <div className="flex justify-between text-sm">
                <span className="text-muted-foreground">Confirmed</span>
                <span className="text-foreground">{confirmedCrash}</span>
              </div>
              <div className="flex justify-between text-sm">
                <span className="text-muted-foreground">False Positives</span>
                <span className="text-foreground">{dismissedCrash}</span>
              </div>
            </div>

            <div className="flex gap-2">
              <Button
                size="sm"
                variant="outline"
                className="flex-1 border-border text-foreground hover:bg-muted"
                onClick={() => handleExportTrainingData('crash')}
              >
                <Download className="w-3 h-3 mr-1" />
                Export
              </Button>
              <Button
                size="sm"
                className="flex-1 bg-amber-600 hover:bg-amber-700 text-white"
                onClick={() => handleStartRetraining('crash')}
                disabled={isRetraining}
              >
                <RefreshCw className="w-3 h-3 mr-1" />
                Retrain
              </Button>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Tabs for different sections */}
      <Tabs value={activeTab} onValueChange={setActiveTab}>
        <TabsList className="grid w-full grid-cols-3 bg-muted">
          <TabsTrigger value="overview" className="data-[state=active]:bg-blue-600 data-[state=active]:text-white">
            Model Performance
          </TabsTrigger>
          <TabsTrigger value="samples" className="data-[state=active]:bg-blue-600 data-[state=active]:text-white">
            Training Samples
          </TabsTrigger>
          <TabsTrigger value="feedback" className="data-[state=active]:bg-blue-600 data-[state=active]:text-white">
            Recent Feedback
          </TabsTrigger>
        </TabsList>

        {/* Model Performance Tab */}
        <TabsContent value="overview" className="space-y-6">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Violence Model Accuracy Trend */}
            <Card className="bg-card/80 backdrop-blur border-border">
              <CardHeader>
                <CardTitle className="flex items-center gap-2 text-card-foreground">
                  <Swords className="w-4 h-4 text-red-600 dark:text-red-400" />
                  Violence Detection Accuracy
                </CardTitle>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={250}>
                  <LineChart data={violenceAccuracyTrend}>
                    <CartesianGrid strokeDasharray="3 3" stroke={theme === 'dark' ? '#374151' : '#e5e7eb'} />
                    <XAxis dataKey="week" stroke={theme === 'dark' ? '#9ca3af' : '#6b7280'} />
                    <YAxis stroke={theme === 'dark' ? '#9ca3af' : '#6b7280'} domain={[90, 100]} />
                    <Tooltip
                      contentStyle={{
                        backgroundColor: theme === 'dark' ? '#1f2937' : '#ffffff',
                        border: `1px solid ${theme === 'dark' ? '#374151' : '#e5e7eb'}`,
                        borderRadius: '8px',
                        color: theme === 'dark' ? '#ffffff' : '#1f2937',
                      }}
                    />
                    <Line type="monotone" dataKey="accuracy" stroke="#ef4444" strokeWidth={3} name="Accuracy %" />
                  </LineChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>

            {/* Crash Model Accuracy Trend */}
            <Card className="bg-card/80 backdrop-blur border-border">
              <CardHeader>
                <CardTitle className="flex items-center gap-2 text-card-foreground">
                  <Car className="w-4 h-4 text-amber-600 dark:text-amber-400" />
                  Car Crash Detection Accuracy
                </CardTitle>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={250}>
                  <LineChart data={crashAccuracyTrend}>
                    <CartesianGrid strokeDasharray="3 3" stroke={theme === 'dark' ? '#374151' : '#e5e7eb'} />
                    <XAxis dataKey="week" stroke={theme === 'dark' ? '#9ca3af' : '#6b7280'} />
                    <YAxis stroke={theme === 'dark' ? '#9ca3af' : '#6b7280'} domain={[95, 100]} />
                    <Tooltip
                      contentStyle={{
                        backgroundColor: theme === 'dark' ? '#1f2937' : '#ffffff',
                        border: `1px solid ${theme === 'dark' ? '#374151' : '#e5e7eb'}`,
                        borderRadius: '8px',
                        color: theme === 'dark' ? '#ffffff' : '#1f2937',
                      }}
                    />
                    <Line type="monotone" dataKey="accuracy" stroke="#f59e0b" strokeWidth={3} name="Accuracy %" />
                  </LineChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">

            {/* Violence Model Feedback Distribution */}
            <Card className="bg-card/80 backdrop-blur border-border">
              <CardHeader>
                <CardTitle className="flex items-center gap-2 text-card-foreground">
                  <Swords className="w-4 h-4 text-red-600 dark:text-red-400" />
                  Violence Detection Feedback
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                {violenceFeedbackStats.map((stat) => (
                  <div key={stat.category} className="space-y-2">
                    <div className="flex items-center justify-between text-sm">
                      <span className="text-foreground">{stat.category}</span>
                      <span className="text-muted-foreground">
                        {stat.value} ({stat.percentage}%)
                      </span>
                    </div>
                    <Progress value={stat.percentage} className="h-2 bg-muted" />
                  </div>
                ))}
              </CardContent>
            </Card>

            {/* Crash Model Feedback Distribution */}
            <Card className="bg-card/80 backdrop-blur border-border">
              <CardHeader>
                <CardTitle className="flex items-center gap-2 text-card-foreground">
                  <Car className="w-4 h-4 text-amber-600 dark:text-amber-400" />
                  Car Crash Detection Feedback
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                {crashFeedbackStats.map((stat) => (
                  <div key={stat.category} className="space-y-2">
                    <div className="flex items-center justify-between text-sm">
                      <span className="text-foreground">{stat.category}</span>
                      <span className="text-muted-foreground">
                        {stat.value} ({stat.percentage}%)
                      </span>
                    </div>
                    <Progress value={stat.percentage} className="h-2 bg-muted" />
                  </div>
                ))}
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        {/* Training Samples Tab */}
        <TabsContent value="samples" className="space-y-6">
          <Card className="bg-card/80 backdrop-blur border-border">
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-card-foreground">
                <Database className="w-5 h-5 text-blue-600 dark:text-blue-400" />
                Training Samples Dataset ({totalSamples} samples)
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                {trainingSamples.map((sample) => (
                  <div
                    key={sample.id}
                    className="flex items-center justify-between p-4 rounded-lg bg-muted/50 border border-border"
                  >
                    <div className="flex items-center gap-4">
                      <div
                        className={`w-10 h-10 rounded-lg flex items-center justify-center ${sample.type === 'violence'
                          ? 'bg-red-100 text-red-600 border-red-200 dark:bg-red-500/20 dark:text-red-400 dark:border-red-500/30'
                          : 'bg-amber-100 text-amber-600 border-amber-200 dark:bg-amber-500/20 dark:text-amber-400 dark:border-amber-500/30'
                          }`}
                      >
                        {sample.type === 'violence' ? (
                          <Swords className="w-5 h-5" />
                        ) : (
                          <Car className="w-5 h-5" />
                        )}
                      </div>
                      <div>
                        <div className="flex items-center gap-2 mb-1">
                          <span className="text-foreground">Sample #{sample.id}</span>
                          <Badge
                            variant="outline"
                            className={`text-xs ${sample.type === 'violence'
                              ? 'border-red-500/50 text-red-600 dark:text-red-400'
                              : 'border-amber-500/50 text-amber-600 dark:text-amber-400'
                              }`}
                          >
                            {sample.type}
                          </Badge>
                          <Badge className={getConfidenceBadgeStyle(
                            sample.confidence >= 85 ? 'high' : sample.confidence >= 60 ? 'medium' : 'low'
                          )}>
                            {sample.confidence}% confidence
                          </Badge>
                        </div>
                        <div className="text-sm text-muted-foreground">
                          {sample.location} • {sample.timestamp}
                        </div>
                      </div>
                    </div>
                    <div className="flex items-center gap-4">
                      <Badge
                        className={
                          sample.usedInTraining
                            ? 'bg-green-100 text-green-800 border-green-200 dark:bg-green-500/20 dark:text-green-400 dark:border-green-500/30'
                            : 'bg-gray-100 text-gray-800 border-gray-200 dark:bg-gray-500/20 dark:text-gray-400 dark:border-gray-500/30'
                        }
                      >
                        {sample.usedInTraining ? 'Used in Training' : 'Pending Use'}
                      </Badge>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Recent Feedback Tab */}
        <TabsContent value="feedback">
          <Card className="bg-card/80 backdrop-blur border-border">
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-card-foreground">
                <CheckCircle className="w-5 h-5 text-green-600 dark:text-green-400" />
                Recent User Feedback
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                {feedbackData.map((feedback) => (
                  <div
                    key={feedback.id}
                    className="flex items-center justify-between p-4 rounded-lg bg-muted/50 border border-border"
                  >
                    <div className="flex items-center gap-4">
                      <div
                        className={`w-10 h-10 rounded-lg flex items-center justify-center ${feedback.userFeedback === 'Correct'
                          ? 'bg-green-100 text-green-600 border-green-200 dark:bg-green-500/20 dark:text-green-400 dark:border-green-500/30'
                          : 'bg-red-100 text-red-600 border-red-200 dark:bg-red-500/20 dark:text-red-400 dark:border-red-500/30'
                          }`}
                      >
                        {feedback.userFeedback === 'Correct' ? (
                          <CheckCircle className="w-5 h-5" />
                        ) : (
                          <XCircle className="w-5 h-5" />
                        )}
                      </div>
                      <div>
                        <div className="flex items-center gap-2 mb-1">
                          <span className="text-foreground">{feedback.incidentId}</span>
                          <Badge
                            variant="outline"
                            className={`text-xs ${feedback.type === 'violence'
                              ? 'border-red-500/50 text-red-600 dark:text-red-400'
                              : 'border-amber-500/50 text-amber-600 dark:text-amber-400'
                              }`}
                          >
                            {feedback.aiPrediction}
                          </Badge>
                          <Badge className={getConfidenceBadgeStyle(feedback.confidenceLevel)}>
                            {feedback.confidence}% {feedback.confidenceLevel}
                          </Badge>
                        </div>
                        <div className="text-sm text-muted-foreground">{feedback.timestamp}</div>
                      </div>
                    </div>
                    <div className="flex items-center gap-4">
                      <div className="text-right">
                        <div className="text-sm text-muted-foreground">Model</div>
                        <Badge
                          className={
                            feedback.model === 'violence'
                              ? 'bg-red-100 text-red-800 border-red-200 dark:bg-red-500/20 dark:text-red-400 dark:border-red-500/30'
                              : 'bg-amber-100 text-amber-800 border-amber-200 dark:bg-amber-500/20 dark:text-amber-400 dark:border-amber-500/30'
                          }
                        >
                          {feedback.model}
                        </Badge>
                      </div>
                      <Badge
                        className={
                          feedback.userFeedback === 'Correct'
                            ? 'bg-green-100 text-green-800 border-green-200 dark:bg-green-500/20 dark:text-green-400 dark:border-green-500/30'
                            : 'bg-red-100 text-red-800 border-red-200 dark:bg-red-500/20 dark:text-red-400 dark:border-red-500/30'
                        }
                      >
                        {feedback.userFeedback}
                      </Badge>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}