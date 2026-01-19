import { Card } from './ui/card';
import { Badge } from './ui/badge';
import { vigilClasses } from './vigil-theme';
import {
  Camera,
  AlertCircle,
  Clock,
  TrendingUp,
  TrendingDown,
  Radio,
  Shield,
  Activity,
  Users,
} from 'lucide-react';
import { motion } from 'motion/react';

interface CompactStatsBarProps {
  role: 'admin' | 'officer' | 'security-authority';
}

export function CompactStatsBar({ role }: CompactStatsBarProps) {
  const stats = [
    {
      label: 'Cameras Online',
      value: '243/247',
      percentage: 98,
      icon: Camera,
      color: 'text-cyan-500',
      bgColor: 'bg-cyan-500/10',
      trend: '+2',
      trendUp: true,
    },
    {
      label: 'Active Alerts',
      value: '8',
      icon: AlertCircle,
      color: 'text-red-500',
      bgColor: 'bg-red-500/10',
      trend: '-3',
      trendUp: false,
    },
    {
      label: 'Response Time',
      value: '2.4m',
      icon: Clock,
      color: 'text-green-500',
      bgColor: 'bg-green-500/10',
      trend: '-0.3m',
      trendUp: false,
    },
    {
      label: 'AI Accuracy',
      value: '94.2%',
      icon: Activity,
      color: 'text-purple-500',
      bgColor: 'bg-purple-500/10',
      trend: '+1.2%',
      trendUp: true,
    },
  ];

  // Add role-specific stats
  if (role === 'officer') {
    stats.push({
      label: 'Personnel',
      value: '8/12',
      icon: Users,
      color: 'text-blue-500',
      bgColor: 'bg-blue-500/10',
      trend: 'On Duty',
      trendUp: true,
    });
  }

  if (role === 'admin') {
    stats.push({
      label: 'System Health',
      value: '99.8%',
      icon: Shield,
      color: 'text-emerald-500',
      bgColor: 'bg-emerald-500/10',
      trend: 'Optimal',
      trendUp: true,
    });
  }

  return (
    <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-2 sm:gap-3">
      {stats.map((stat, index) => {
        const Icon = stat.icon;
        const TrendIcon = stat.trendUp ? TrendingUp : TrendingDown;
        
        return (
          <motion.div
            key={stat.label}
            initial={{ y: -10, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            transition={{ delay: index * 0.05 }}
          >
            <Card
              className={`${vigilClasses.card} hover:border-cyan-500/30 transition-all duration-300 group cursor-pointer`}
            >
              <div className="p-3 sm:p-4">
                <div className="flex items-center justify-between mb-2">
                  <div className={`p-2 rounded-lg ${stat.bgColor}`}>
                    <Icon className={`w-4 h-4 ${stat.color}`} />
                  </div>
                  {stat.trend && (
                    <Badge
                      variant="secondary"
                      className={`text-xs ${
                        stat.trendUp
                          ? 'bg-green-500/20 text-green-600 dark:text-green-400'
                          : 'bg-amber-500/20 text-amber-600 dark:text-amber-400'
                      } border-0`}
                    >
                      {typeof stat.trend === 'string' && !stat.trend.startsWith('+') && !stat.trend.startsWith('-') ? (
                        stat.trend
                      ) : (
                        <>
                          <TrendIcon className="w-3 h-3 mr-1" />
                          {stat.trend}
                        </>
                      )}
                    </Badge>
                  )}
                </div>
                <div className={`text-xl sm:text-2xl ${vigilClasses.textPrimary} mb-1`}>
                  {stat.value}
                </div>
                <div className={`text-xs ${vigilClasses.textSecondary}`}>{stat.label}</div>
                {stat.percentage !== undefined && (
                  <div className="mt-2">
                    <div className="h-1 bg-gray-200 dark:bg-gray-800 rounded-full overflow-hidden">
                      <motion.div
                        initial={{ width: 0 }}
                        animate={{ width: `${stat.percentage}%` }}
                        transition={{ delay: index * 0.1, duration: 0.5 }}
                        className={`h-full ${stat.color.replace('text-', 'bg-')}`}
                      />
                    </div>
                  </div>
                )}
              </div>
            </Card>
          </motion.div>
        );
      })}
    </div>
  );
}
