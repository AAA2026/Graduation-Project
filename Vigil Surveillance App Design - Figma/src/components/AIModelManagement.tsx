import { useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';
import { Button } from './ui/button';
import { Badge } from './ui/badge';
import { Progress } from './ui/progress';
import { toast } from 'sonner@2.0.3';
import {
  Brain,
  Upload,
  Download,
  Play,
  Pause,
  RefreshCw,
  TrendingUp,
  TrendingDown,
  Activity,
  FileText,
  Clock,
  CheckCircle,
  AlertTriangle,
  Swords,
  Car,
  Users,
} from 'lucide-react';
import { useTheme } from './ThemeProvider';

interface AIModel {
  id: string;
  name: string;
  type: 'violence' | 'crash';
  version: string;
  status: 'active' | 'training' | 'inactive' | 'updating';
  accuracy: number;
  accuracyTrend: 'up' | 'down' | 'stable';
  deployedDate: string;
  lastTraining: string;
  totalPredictions: number;
  falsePositives: number;
  falseNegatives: number;
  trainingDataSize: number;
}

export function AIModelManagement() {
  const [models] = useState<AIModel[]>([
    {
      id: '1',
      name: 'Violence Detection Model',
      type: 'violence',
      version: 'v4.2.1',
      status: 'active',
      accuracy: 97.3,
      accuracyTrend: 'up',
      deployedDate: '2024-11-01',
      lastTraining: '2 days ago',
      totalPredictions: 45320,
      falsePositives: 234,
      falseNegatives: 128,
      trainingDataSize: 125000,
    },
    {
      id: '2',
      name: 'Car Crash Detection Model',
      type: 'crash',
      version: 'v3.8.2',
      status: 'active',
      accuracy: 98.9,
      accuracyTrend: 'stable',
      deployedDate: '2024-10-28',
      lastTraining: '5 days ago',
      totalPredictions: 23890,
      falsePositives: 189,
      falseNegatives: 245,
      trainingDataSize: 98000,
    },
  ]);
    // People Counter Model removed

  const [retrainingProgress] = useState(42);
  const { theme } = useTheme();

  const handleRetrainModel = (modelName: string) => {
    toast.success(`Retraining process started for ${modelName}`);
  };

  const handleUpdateModel = (modelName: string) => {
    toast.success(`Model update initiated for ${modelName}`);
  };

  const handleExportModel = (modelName: string) => {
    toast.success(`Exporting ${modelName}...`);
  };

  const getStatusBadge = (status: string) => {
    switch (status) {
      case 'active':
        return <Badge className="bg-green-100 text-green-800 border-green-200 dark:bg-green-500/20 dark:text-green-400 dark:border-green-500/30"><CheckCircle className="w-3 h-3 mr-1" />Active</Badge>;
      case 'training':
        return <Badge className="bg-blue-100 text-blue-800 border-blue-200 dark:bg-blue-500/20 dark:text-blue-400 dark:border-blue-500/30"><RefreshCw className="w-3 h-3 mr-1 animate-spin" />Training</Badge>;
      case 'updating':
        return <Badge className="bg-amber-100 text-amber-800 border-amber-200 dark:bg-amber-500/20 dark:text-amber-400 dark:border-amber-500/30"><Upload className="w-3 h-3 mr-1" />Updating</Badge>;
      case 'inactive':
        return <Badge className="bg-gray-100 text-gray-800 border-gray-200 dark:bg-gray-500/20 dark:text-gray-400 dark:border-gray-500/30"><Pause className="w-3 h-3 mr-1" />Inactive</Badge>;
      default:
        return <Badge>Unknown</Badge>;
    }
  };

  const getModelIcon = (type: 'violence' | 'crash') => {
    switch (type) {
      case 'violence':
        return <Swords className="w-6 h-6 text-red-600 dark:text-red-400" />;
      case 'crash':
        return <Car className="w-6 h-6 text-amber-600 dark:text-amber-400" />;
    }
  };

  const getModelColor = (type: 'violence' | 'crash') => {
    switch (type) {
      case 'violence':
        return {
          bg: 'bg-red-100 dark:bg-red-900/30',
          text: 'text-red-600 dark:text-red-400',
          progress: '#ef4444'
        };
      case 'crash':
        return {
          bg: 'bg-amber-100 dark:bg-amber-900/30',
          text: 'text-amber-600 dark:text-amber-400',
          progress: '#f59e0b'
        };
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
        <div>
          <h2 className="text-2xl font-bold text-foreground mb-1">AI Model Management</h2>
          <p className="text-muted-foreground">
            Update, retrain, and monitor AI detection models
          </p>
        </div>
        <Button className="bg-blue-600 hover:bg-blue-700 text-white">
          <Upload className="w-4 h-4 mr-2" />
          Upload New Model
        </Button>
      </div>

      {/* Overview Stats */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card className="bg-card/80 backdrop-blur border-border">
          <CardContent className="p-6">
            <div className="flex items-center gap-3">
              <div className="p-3 rounded-lg bg-blue-100 dark:bg-blue-900/30">
                <Brain className="w-6 h-6 text-blue-600 dark:text-blue-400" />
              </div>
              <div>
                <p className="text-sm text-muted-foreground">Total Models</p>
                <p className="text-2xl text-foreground">2</p>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card className="bg-card/80 backdrop-blur border-border">
          <CardContent className="p-6">
            <div className="flex items-center gap-3">
              <div className="p-3 rounded-lg bg-green-100 dark:bg-green-900/30">
                <CheckCircle className="w-6 h-6 text-green-600 dark:text-green-400" />
              </div>
              <div>
                <p className="text-sm text-muted-foreground">Active Models</p>
                <p className="text-2xl text-foreground">2</p>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card className="bg-card/80 backdrop-blur border-border">
          <CardContent className="p-6">
            <div className="flex items-center gap-3">
              <div className="p-3 rounded-lg bg-amber-100 dark:bg-amber-900/30">
                <RefreshCw className="w-6 h-6 text-amber-600 dark:text-amber-400" />
              </div>
              <div>
                <p className="text-sm text-muted-foreground">In Training</p>
                <p className="text-2xl text-foreground">0</p>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card className="bg-card/80 backdrop-blur border-border">
          <CardContent className="p-6">
            <div className="flex items-center gap-3">
              <div className="p-3 rounded-lg bg-purple-100 dark:bg-purple-900/30">
                <Activity className="w-6 h-6 text-purple-600 dark:text-purple-400" />
              </div>
              <div>
                <p className="text-sm text-muted-foreground">Avg Accuracy</p>
                <p className="text-2xl text-foreground">98.1%</p>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Model Cards - Only 2 Models */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {models.map((model) => {
          const colors = getModelColor(model.type);
          
          return (
            <Card key={model.id} className="bg-card/80 backdrop-blur border-border">
              <CardHeader>
                <div className="flex items-start justify-between">
                  <div className="flex items-center gap-3">
                    <div className={`p-2 rounded-lg ${colors.bg}`}>
                      {getModelIcon(model.type)}
                    </div>
                    <div>
                      <CardTitle className="text-foreground">{model.name}</CardTitle>
                      <CardDescription className="text-muted-foreground">
                        Version {model.version} • Deployed {model.deployedDate}
                      </CardDescription>
                    </div>
                  </div>
                  {getStatusBadge(model.status)}
                </div>
              </CardHeader>
              <CardContent className="space-y-4">
                {/* Accuracy */}
                <div>
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-sm text-muted-foreground">Model Accuracy</span>
                    <div className="flex items-center gap-2">
                      {model.accuracyTrend === 'up' && <TrendingUp className="w-4 h-4 text-green-500" />}
                      {model.accuracyTrend === 'down' && <TrendingDown className="w-4 h-4 text-red-500" />}
                      <span className="text-foreground font-semibold">{model.accuracy}%</span>
                    </div>
                  </div>
                  <Progress value={model.accuracy} className="h-2 bg-muted" />
                </div>

                {/* Stats Grid */}
                <div className="grid grid-cols-2 gap-4 pt-2">
                  <div className="p-3 rounded-lg bg-muted/50 border border-border">
                    <p className="text-xs text-muted-foreground mb-1">Total Predictions</p>
                    <p className="text-foreground font-semibold">{model.totalPredictions.toLocaleString()}</p>
                  </div>
                  <div className="p-3 rounded-lg bg-muted/50 border border-border">
                    <p className="text-xs text-muted-foreground mb-1">Training Data</p>
                    <p className="text-foreground font-semibold">{model.trainingDataSize.toLocaleString()}</p>
                  </div>
                  <div className="p-3 rounded-lg bg-muted/50 border border-border">
                    <p className="text-xs text-muted-foreground mb-1">False Positives</p>
                    <p className="text-amber-600 dark:text-amber-400 font-semibold">{model.falsePositives}</p>
                  </div>
                  <div className="p-3 rounded-lg bg-muted/50 border border-border">
                    <p className="text-xs text-muted-foreground mb-1">False Negatives</p>
                    <p className="text-red-600 dark:text-red-400 font-semibold">{model.falseNegatives}</p>
                  </div>
                </div>

                {/* Last Training */}
                <div className="flex items-center gap-2 text-sm text-muted-foreground">
                  <Clock className="w-4 h-4" />
                  <span>Last trained {model.lastTraining}</span>
                </div>

                {/* Training Progress (if training) */}
                {model.status === 'training' && (
                  <div className="p-3 rounded-lg bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800">
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-sm text-blue-700 dark:text-blue-400">Training in progress...</span>
                      <span className="text-sm text-blue-700 dark:text-blue-400">{retrainingProgress}%</span>
                    </div>
                    <Progress value={retrainingProgress} className="h-2 bg-blue-200 dark:bg-blue-500/20" />
                  </div>
                )}

                {/* Actions */}
                <div className="flex gap-2 pt-2">
                  <Button
                    size="sm"
                    variant="outline"
                    className="flex-1 border-border text-foreground hover:bg-muted"
                    onClick={() => handleRetrainModel(model.name)}
                    disabled={model.status === 'training'}
                  >
                    <RefreshCw className="w-4 h-4 mr-2" />
                    Retrain
                  </Button>
                  <Button
                    size="sm"
                    variant="outline"
                    className="flex-1 border-border text-foreground hover:bg-muted"
                    onClick={() => handleUpdateModel(model.name)}
                    disabled={model.status === 'training'}
                  >
                    <Upload className="w-4 h-4 mr-2" />
                    Update
                  </Button>
                  <Button
                    size="sm"
                    variant="outline"
                    className="border-border text-foreground hover:bg-muted"
                    onClick={() => handleExportModel(model.name)}
                  >
                    <Download className="w-4 h-4" />
                  </Button>
                </div>
              </CardContent>
            </Card>
          );
        })}
      </div>

      {/* Training History */}
      <Card className="bg-card/80 backdrop-blur border-border">
        <CardHeader>
          <CardTitle className="text-foreground">Recent Training Sessions</CardTitle>
          <CardDescription className="text-muted-foreground">History of model training and updates</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-3">
            {[
              { model: 'Violence Detection', action: 'Retrained', date: '2 days ago', result: 'Success', accuracy: '+1.2%' },
              { model: 'Car Crash Detection', action: 'Retrained', date: '5 days ago', result: 'Success', accuracy: '+0.5%' },
              { model: 'Violence Detection', action: 'Updated', date: '1 week ago', result: 'Success', accuracy: '+2.1%' },
              { model: 'Car Crash Detection', action: 'Updated', date: '2 weeks ago', result: 'Success', accuracy: '+1.8%' },
            ].map((session, index) => (
              <div
                key={index}
                className="flex items-center justify-between p-3 rounded-lg bg-muted/50 border border-border"
              >
                <div className="flex items-center gap-3">
                  <div className={`p-2 rounded-lg ${session.result === 'Success' ? 'bg-green-100 dark:bg-green-900/30' : 'bg-blue-100 dark:bg-blue-900/30'}`}>
                    {session.result === 'Success' ? (
                      <CheckCircle className="w-4 h-4 text-green-600 dark:text-green-400" />
                    ) : (
                      <RefreshCw className="w-4 h-4 text-blue-600 dark:text-blue-400 animate-spin" />
                    )}
                  </div>
                  <div>
                    <p className="text-foreground">{session.model}</p>
                    <p className="text-sm text-muted-foreground">
                      {session.action} • {session.date}
                    </p>
                  </div>
                </div>
                <div className="text-right">
                  <Badge className={
                    session.result === 'Success' 
                      ? 'bg-green-100 text-green-800 border-green-200 dark:bg-green-500/20 dark:text-green-400 dark:border-green-500/30'
                      : 'bg-blue-100 text-blue-800 border-blue-200 dark:bg-blue-500/20 dark:text-blue-400 dark:border-blue-500/30'
                  }>
                    {session.result}
                  </Badge>
                  {session.accuracy !== 'TBD' && (
                    <p className="text-sm text-green-600 dark:text-green-400 mt-1">{session.accuracy}</p>
                  )}
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  );
}