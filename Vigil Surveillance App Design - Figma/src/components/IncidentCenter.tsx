import { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Button } from './ui/button';
import { Badge } from './ui/badge';
import { ScrollArea } from './ui/scroll-area';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './ui/tabs';
import { toast } from 'sonner';
import { useTheme } from './ThemeProvider';
import { Eye, X, AlertCircle, Car, Users, Swords } from 'lucide-react';
import { useRealtimeIncidents } from '../hooks/useRealtimeIncidents';
import { IncidentDetailModal } from './IncidentDetailModal';

export function IncidentCenter() {
  const { theme } = useTheme();
  const {
    incidents,
    resolveIncident,
    dismissIncident,
    soundEnabled,
    toggleSound,
  } = useRealtimeIncidents();
  const [modalIncidentId, setModalIncidentId] = useState<string | null>(null);

  // Only show violence/crash as notifications
  const notificationIncidents = incidents.filter(
    (inc: any) => inc.type === 'violence' || inc.type === 'crash'
  );

  return (
    <Card className="w-full max-w-2xl mx-auto bg-background border shadow-lg">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <AlertCircle className="w-5 h-5 text-red-500" />
          Incidents & Notifications
          <Button
            variant="ghost"
            size="sm"
            className="ml-auto"
            onClick={toggleSound}
            title={soundEnabled ? 'Disable alert sound' : 'Enable alert sound'}
          >
            {soundEnabled ? '🔊' : '🔇'}
          </Button>
        </CardTitle>
      </CardHeader>
      <CardContent>
        <ScrollArea className="h-96 pr-2">
          {notificationIncidents.length === 0 && (
            <div className="text-center text-gray-500 py-8">No incidents or notifications.</div>
          )}
          {notificationIncidents.map((inc: any) => (
            <div
              key={inc.id}
              className={`flex items-center gap-3 p-3 mb-2 rounded-lg border ${inc.type === 'violence' ? 'border-red-400 bg-red-50 dark:bg-red-900/20' : 'border-yellow-400 bg-yellow-50 dark:bg-yellow-900/20'}`}
            >
              <div className="flex-shrink-0">
                {inc.type === 'violence' ? (
                  <Swords className="w-6 h-6 text-red-500" />
                ) : (
                  <Car className="w-6 h-6 text-yellow-500" />
                )}
              </div>
              <div className="flex-1 min-w-0">
                <div className="font-semibold text-sm text-gray-900 dark:text-white">
                  {inc.type === 'violence'
                    ? (inc.people_count > 0 ? `Violence detected (${inc.people_count} people)` : `Violence detected`)
                    : 'Crash detected'}
                  <span className="ml-2 text-xs text-gray-500">{inc.cameraId}</span>
                </div>
                <div className="text-xs text-gray-600 dark:text-gray-300">
                  {inc.location || 'Unknown location'}
                </div>
                <div className="text-xs text-gray-400 mt-1">
                  {new Date(inc.timestamp).toLocaleString()}
                </div>
              </div>
              <Button
                size="icon"
                variant="ghost"
                className="text-gray-400 hover:text-red-500"
                onClick={() => dismissIncident(inc.id)}
                title="Dismiss"
              >
                <X className="w-4 h-4" />
              </Button>
              <Button
                size="sm"
                className="ml-1 bg-blue-600 hover:bg-blue-700 text-white"
                onClick={() => setModalIncidentId(inc.id)}
              >
                <Eye className="w-4 h-4 mr-1" /> View
              </Button>
            </div>
          ))}
        </ScrollArea>
      </CardContent>
      {modalIncidentId && (
        <IncidentDetailModal
          incident={incidents.find((i: any) => i.id === modalIncidentId) || null}
          isOpen={!!modalIncidentId}
          onClose={() => setModalIncidentId(null)}
        />
      )}
    </Card>
  );
}
