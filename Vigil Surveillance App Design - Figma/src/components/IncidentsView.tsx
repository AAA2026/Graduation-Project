import { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Badge } from './ui/badge';
import { Button } from './ui/button';
import { vigilClasses } from './vigil-theme';
import { useTheme } from './ThemeProvider';
import { Input } from './ui/input';
import {
  AlertCircle,
  Car,
  Users,
  Clock,
  MapPin,
  Eye,
  CheckCircle,
  XCircle,
  Play,
  Filter,
  Search,
  FileText,
  Download,
  Maximize2,
  Volume2,
  VolumeX,
  UserCheck,
  UserPlus,
} from 'lucide-react';
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription, DialogFooter } from './ui/dialog';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './ui/select';
import { motion, AnimatePresence } from 'motion/react';
import Breadcrumb, { BreadcrumbItem } from './Breadcrumb';
import { exportIncidentsToCSV, exportIncidentsToPDF } from '../utils/exportUtils';
import { toast } from 'sonner';
import { SECURITY_PERSONNEL } from '../data/roster';

import type { Incident } from '../hooks/useRealtimeIncidents';

interface IncidentsViewProps {
  role: 'admin' | 'officer' | 'security';
  incidents?: Incident[];
  onIncidentClick?: (id: string) => void;
  onResolve?: (id: string) => void;
  onDismiss?: (id: string) => void;
  onDismissAll?: () => void;
  soundEnabled?: boolean;
  onToggleSound?: () => void;
  title?: string;
  onSelfDispatch?: (id: string) => void;
  onDispatch?: (id: string, securityId: string) => void;
}

export function IncidentsView({
  role,
  incidents: externalIncidents,
  onIncidentClick,
  onResolve,
  onDismiss,
  onDismissAll,
  soundEnabled,
  onToggleSound,
  title = "Incidents",
  onSelfDispatch,
  onDispatch,
}: IncidentsViewProps) {
  const [filter, setFilter] = useState<'all' | 'active' | 'resolved'>('active');
  const [searchQuery, setSearchQuery] = useState('');
  const [showAdvancedFilters, setShowAdvancedFilters] = useState(false);
  const [severityFilter, setSeverityFilter] = useState<'all' | 'critical' | 'high' | 'medium' | 'low'>('all');
  const { theme } = useTheme();

  // Only use real incidents from props; never show mock/demo data
  const incidents = externalIncidents || [];

  // Dispatch State
  const [dispatchModalOpen, setDispatchModalOpen] = useState(false);
  const [incidentToDispatch, setIncidentToDispatch] = useState<string | null>(null);
  const [selectedOfficer, setSelectedOfficer] = useState<string>('');
  const [dispatchStatus, setDispatchStatus] = useState<Record<string, { officerId: string, officerName: string, time: string }>>({});

  // Mock Security Personnel
  // Imported from shared data

  const handleOpenDispatch = (incidentId: string, e: React.MouseEvent) => {
    e.stopPropagation();
    setIncidentToDispatch(incidentId);
    setDispatchModalOpen(true);
    setSelectedOfficer('');
  };

  const handleConfirmDispatch = () => {
    if (!incidentToDispatch || !selectedOfficer) return;

    const officer = SECURITY_PERSONNEL.find(p => p.id === selectedOfficer);
    if (officer) {
      setDispatchStatus(prev => ({
        ...prev,
        [incidentToDispatch]: {
          officerId: officer.id,
          officerName: officer.name,
          time: new Date().toLocaleTimeString()
        }
      }));
      // Call backend dispatch
      onDispatch?.(incidentToDispatch, officer.id);

      toast.success(`Dispatched ${officer.name} to incident ${incidentToDispatch}`);
    }
    setDispatchModalOpen(false);
    setIncidentToDispatch(null);
  };

  const filteredIncidents = incidents.filter((inc) => {
    // Status Filter
    if (filter === 'active') {
      // Only show strictly active (unhandled) incidents. 
      // Dispatched/Acknowledged go to "Active Dispatches" view.
      if (inc.status !== 'active') return false;
    } else if (filter === 'resolved') {
      if (inc.status !== 'resolved') return false;
    } else if (filter !== 'all' && inc.status !== filter) {
      // Fallback for other potential statuses
      return false;
    }

    // Search Filter
    if (searchQuery) {
      const query = searchQuery.toLowerCase();
      const matchesId = inc.id.toLowerCase().includes(query);
      const matchesLocation = inc.location.toLowerCase().includes(query);
      const matchesType = inc.type.toLowerCase().includes(query);
      if (!matchesId && !matchesLocation && !matchesType) return false;
    }

    // Severity Filter
    if (severityFilter !== 'all' && inc.severity !== severityFilter) return false;

    return true;
  }).slice(0, 50);

  const getTypeIcon = (type: string) => {
    if (type === 'violence') return Users;
    return Car;
  };

  const getSeverityColor = (severity: string) => {
    if (severity === 'critical') return 'bg-red-600 text-white';
    if (severity === 'high') return 'bg-orange-600 text-white';
    if (severity === 'medium') return 'bg-amber-600 text-white';
    return 'bg-blue-600 text-white';
  };

  const activeCount = incidents.filter((i) => i.status === 'active').length;
  const resolvedCount = incidents.filter((i) => i.status === 'resolved').length;

  // Export handlers
  const handleExportCSV = () => {
    try {
      const exportData = filteredIncidents.map(inc => ({
        id: inc.id,
        timestamp: inc.timestamp,
        cameraId: inc.cameraId,
        cameraName: inc.cameraId, // Using cameraId as name for now
        type: inc.type,
        confidence: inc.confidence / 100,
        location: inc.location,
        status: inc.status,
        aiModel: 'YOLO-v8',
      }));
      exportIncidentsToCSV(exportData as any);
      toast.success('Incidents exported to CSV successfully');
    } catch (error) {
      toast.error('Failed to export CSV');
      console.error(error);
    }
  };

  const handleExportPDF = () => {
    try {
      const exportData = filteredIncidents.map(inc => ({
        id: inc.id,
        timestamp: inc.timestamp,
        cameraId: inc.cameraId,
        cameraName: inc.cameraId,
        type: inc.type,
        confidence: inc.confidence / 100,
        location: inc.location,
        status: inc.status,
        aiModel: 'YOLO-v8',
      }));
      exportIncidentsToPDF(exportData as any);
      toast.success('Incidents exported to PDF successfully');
    } catch (error) {
      toast.error('Failed to export PDF');
      console.error(error);
    }
  };

  return (
    <div className="space-y-4">
      {/* Dispatch Modal */}
      <Dialog open={dispatchModalOpen} onOpenChange={(open: boolean) => !open && setDispatchModalOpen(false)}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Dispatch Security Personnel</DialogTitle>
            <DialogDescription>
              Select an available security officer to respond to Incident {incidentToDispatch}.
            </DialogDescription>
          </DialogHeader>
          <div className="py-4">
            <label className="text-sm font-medium mb-2 block">Available Officers</label>
            <Select value={selectedOfficer} onValueChange={setSelectedOfficer}>
              <SelectTrigger>
                <SelectValue placeholder="Select Officer" />
              </SelectTrigger>
              <SelectContent>
                {SECURITY_PERSONNEL.map(officer => (
                  <SelectItem key={officer.id} value={officer.id} disabled={officer.status === 'Busy'}>
                    <span className="flex items-center justify-between w-full min-w-[200px]">
                      <span>{officer.name} ({officer.location})</span>
                      <Badge variant={officer.status === 'Available' ? 'outline' : 'secondary'} className="ml-2">
                        {officer.status}
                      </Badge>
                    </span>
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setDispatchModalOpen(false)}>Cancel</Button>
            <Button onClick={handleConfirmDispatch} disabled={!selectedOfficer} className="bg-blue-600 hover:bg-blue-700 text-white">
              Confirm Dispatch
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
      {/* Incidents Tab Header */}
      <div className="flex items-center gap-2">
        <AlertCircle className="w-7 h-7 text-blue-600 dark:text-blue-400" />
        <h1 className="text-2xl font-bold tracking-tight">{title}</h1>
      </div>

      {/* Header with Stats */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <Card className={`${vigilClasses.card} border-2 ${filter === 'active' ? 'border-red-500' : vigilClasses.border}`}>
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className={`text-sm ${vigilClasses.textMuted}`}>Active Incidents</p>
                <div className="flex items-baseline gap-2 mt-1">
                  <p className="text-3xl text-red-400">{Math.min(filteredIncidents.length, 50)}</p>
                  <p className="text-sm text-muted-foreground">/ {activeCount} total</p>
                </div>
              </div>
              <div className="w-12 h-12 bg-red-500/20 rounded-full flex items-center justify-center">
                <AlertCircle className="w-6 h-6 text-red-400" />
              </div>
            </div>
          </CardContent>
        </Card>

        <Card className={`${vigilClasses.card} border-2 ${filter === 'resolved' ? 'border-green-500' : vigilClasses.border}`}>
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className={`text-sm ${vigilClasses.textMuted}`}>Resolved Today</p>
                <p className="text-3xl text-green-400 mt-1">{resolvedCount}</p>
              </div>
              <div className="w-12 h-12 bg-green-500/20 rounded-full flex items-center justify-center">
                <CheckCircle className="w-6 h-6 text-green-400" />
              </div>
            </div>
          </CardContent>
        </Card>

        <Card className={`${vigilClasses.card} border-2 ${vigilClasses.border}`}>
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className={`text-sm ${vigilClasses.textMuted}`}>Avg Response Time</p>
                <p className="text-3xl text-cyan-400 dark:text-cyan-400 mt-1">2.3m</p>
              </div>
              <div className="w-12 h-12 bg-cyan-500/20 rounded-full flex items-center justify-center">
                <Clock className="w-6 h-6 text-cyan-400" />
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Filters Toolbar */}
      <Card className={vigilClasses.card}>
        <CardContent className="p-4">
          <div className="flex flex-col gap-4">
            <div className="flex items-center justify-between flex-wrap gap-3">
              <div className="flex items-center gap-2">
                <Button
                  variant={filter === 'all' ? 'default' : 'outline'}
                  size="sm"
                  onClick={() => setFilter('all')}
                >
                  All ({incidents.length})
                </Button>
                <Button
                  variant={filter === 'active' ? 'default' : 'outline'}
                  size="sm"
                  onClick={() => setFilter('active')}
                  className={filter === 'active' ? 'bg-red-600 hover:bg-red-700' : ''}
                >
                  Active ({activeCount > 50 ? '50+' : activeCount})
                </Button>
                <Button
                  variant={filter === 'resolved' ? 'default' : 'outline'}
                  size="sm"
                  onClick={() => setFilter('resolved')}
                  className={filter === 'resolved' ? 'bg-green-600 hover:bg-green-700' : ''}
                >
                  Resolved Today ({resolvedCount})
                </Button>
              </div>

              <div className="flex items-center gap-2 flex-wrap">
                <div className="relative w-64">
                  <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-500" />
                  <Input
                    placeholder="Search ID, location, type..."
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    className="pl-8 h-9 bg-background/50"
                  />
                </div>
                <Button
                  variant={showAdvancedFilters ? "default" : "outline"}
                  size="sm"
                  onClick={() => setShowAdvancedFilters(!showAdvancedFilters)}
                  className={showAdvancedFilters ? "bg-blue-600" : ""}
                >
                  <Filter className="w-4 h-4 mr-2" />
                  Filter
                </Button>
                <Button variant="outline" size="sm" onClick={handleExportCSV}>
                  <Download className="w-4 h-4 mr-2" />
                  CSV
                </Button>
                <Button variant="outline" size="sm" onClick={handleExportPDF}>
                  <FileText className="w-4 h-4 mr-2" />
                  PDF
                </Button>
                {role === 'admin' && onDismissAll && (
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={onDismissAll}
                    className="text-red-500 border-red-200 hover:bg-red-50 dark:border-red-900/30 dark:hover:bg-red-950/20"
                  >
                    <XCircle className="w-4 h-4 mr-2" />
                    Dismiss All
                  </Button>
                )}
                {onToggleSound && (
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={onToggleSound}
                    className={soundEnabled ? "text-blue-600 border-blue-200" : "text-gray-500 border-gray-200"}
                  >
                    {soundEnabled ? <Volume2 className="w-4 h-4 mr-2" /> : <VolumeX className="w-4 h-4 mr-2" />}
                  </Button>
                )}
              </div>
            </div>

            {/* Advanced Filters Panel */}
            <AnimatePresence>
              {showAdvancedFilters && (
                <motion.div
                  initial={{ height: 0, opacity: 0 }}
                  animate={{ height: 'auto', opacity: 1 }}
                  exit={{ height: 0, opacity: 0 }}
                  className="pt-2 border-t border-gray-100 dark:border-gray-800"
                >
                  <div className="flex items-center gap-4">
                    <span className="text-sm font-medium text-muted-foreground">Severity:</span>
                    <div className="flex gap-2">
                      {['all', 'critical', 'high', 'medium', 'low'].map((sev) => (
                        <Button
                          key={sev}
                          variant={severityFilter === sev ? "secondary" : "ghost"}
                          size="sm"
                          onClick={() => setSeverityFilter(sev as any)}
                          className="capitalize h-7 text-xs"
                        >
                          {sev}
                        </Button>
                      ))}
                    </div>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        </CardContent>
      </Card>

      {/* Incidents List */}
      <div className="space-y-3">
        {filteredIncidents.length === 0 ? (
          <Card className={vigilClasses.card}>
            <CardContent className="p-12 text-center text-muted-foreground">
              <Search className="w-12 h-12 mx-auto mb-4 opacity-50" />
              <p className="text-lg font-medium">No incidents matched your search</p>
              <p className="text-sm">Try adjusting your filters or search query</p>
              <Button variant="link" onClick={() => { setSearchQuery(''); setFilter('all'); setSeverityFilter('all'); }}>
                Clear all filters
              </Button>
            </CardContent>
          </Card>
        ) : (
          filteredIncidents.map((incident, index) => {
            const TypeIcon = getTypeIcon(incident.type);
            const isDispatched = !!dispatchStatus[incident.id];
            const dispatchInfo = dispatchStatus[incident.id];

            // Incident summary
            let summary = '';
            if (incident.type === 'violence') {
              summary = `Violence detected`;
            } else if (incident.type === 'crash') {
              summary = 'Car crash detected';
            } else {
              summary = incident.description || 'Incident detected';
            }
            return (
              <motion.div
                key={incident.id}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: index * 0.05 }}
              >
                <Card
                  className={`${vigilClasses.card} border-l-4 ${incident.status === 'active'
                    ? 'border-l-red-500 bg-white dark:bg-[#13182b] shadow-lg shadow-red-500/5'
                    : 'border-l-green-500 bg-white dark:bg-[#13182b] shadow-lg shadow-green-500/5'
                    }`}
                >
                  <CardContent className="p-4">
                    <div className="flex items-start justify-between gap-4">
                      {/* Left: Icon and Info */}
                      <div className="flex items-start gap-4 flex-1">
                        <div
                          className={`w-12 h-12 rounded-lg flex items-center justify-center ${incident.type === 'violence'
                            ? 'bg-red-500/20 text-red-400'
                            : 'bg-orange-500/20 text-orange-400'
                            }`}
                        >
                          <TypeIcon className="w-6 h-6" />
                        </div>
                        <div>
                          <div className="font-medium">
                            {incident.type === 'violence' ? (incident.people_count ? `Violence (${incident.people_count} people)` : 'Violence') : 'Car Crash'}
                          </div>
                          <div className="text-xs text-muted-foreground">{incident.location}</div>
                        </div>
                        <div className="flex-1 min-w-0">
                          <div className="flex items-center gap-2 flex-wrap mb-2">
                            <h3 className={`font-mono ${vigilClasses.textPrimary}`}>{incident.id}</h3>
                            <Badge className={getSeverityColor(incident.severity)}>
                              {incident.severity.toUpperCase()}
                            </Badge>
                            <Badge
                              className={
                                incident.status === 'active'
                                  ? 'bg-red-500/20 text-red-400 border-red-500/30'
                                  : 'bg-green-500/20 text-green-400 border-green-500/30'
                              }
                            >
                              {incident.status === 'active' ? (
                                <>
                                  <AlertCircle className="w-3 h-3 mr-1 animate-pulse" />
                                  ACTIVE
                                </>
                              ) : (
                                <>
                                  <CheckCircle className="w-3 h-3 mr-1" />
                                  RESOLVED
                                </>
                              )}
                            </Badge>
                            {incident.status === 'dispatched' && (
                              <Badge className="bg-blue-600 text-white border-blue-500 flex items-center gap-1 shadow-lg ml-2">
                                <UserCheck className="w-3 h-3" />
                                {incident.assigned_security ? `On Route: ${incident.assigned_security}` : 'Dispatched'}
                              </Badge>
                            )}
                            {incident.status === 'acknowledged' && (
                              <Badge className="bg-purple-600 text-white border-purple-500 flex items-center gap-1 shadow-lg ml-2">
                                <UserCheck className="w-3 h-3" />
                                {incident.ack_by ? `Ack: ${incident.ack_by}` : 'Acknowledged'}
                              </Badge>
                            )}
                          </div>

                          <p className={vigilClasses.textPrimary + " mb-2"}>{summary}</p>

                          <div className={`flex items-center gap-4 text-sm ${vigilClasses.textMuted}`}>
                            <div className="flex items-center gap-1">
                              <MapPin className="w-4 h-4" />
                              <span>{incident.location}</span>
                            </div>
                            <div className="flex items-center gap-1">
                              <Eye className="w-4 h-4" />
                              <span>{incident.cameraId}</span>
                            </div>
                            <div className="flex items-center gap-1">
                              <Clock className="w-4 h-4" />
                              <span>{incident.timestamp}</span>
                            </div>
                          </div>

                          <div className="mt-2">
                            <div className="flex items-center gap-2">
                              <span className={`text-xs ${vigilClasses.textMuted}`}>AI Confidence:</span>
                              <div className="flex-1 max-w-xs h-1.5 bg-gray-200 dark:bg-gray-800 rounded-full overflow-hidden">
                                <div
                                  className={`h-full ${incident.confidence > 90
                                    ? 'bg-green-500'
                                    : incident.confidence > 80
                                      ? 'bg-cyan-500'
                                      : 'bg-amber-500'
                                    }`}
                                  style={{ width: `${incident.confidence}%` }}
                                />
                              </div>
                              <span className={`text-xs ${vigilClasses.textMuted}`}>{incident.confidence}%</span>
                            </div>
                          </div>
                        </div>
                      </div>

                      {/* Right: Video and Actions */}
                      <div className="flex flex-col gap-2 items-end min-w-[180px]">
                        {/* No video preview/thumbnail shown here */}
                        <Button size="sm" className="bg-cyan-600 hover:bg-cyan-700" onClick={() => onIncidentClick?.(incident.id)}>
                          <Maximize2 className="w-3 h-3 mr-1" />
                          Full Screen
                        </Button>
                        {incident.status === 'active' && (
                          <>
                            <Button size="sm" variant="outline" className="border-green-600 text-green-400 hover:bg-green-950/30" onClick={() => onResolve?.(incident.id)}>
                              <CheckCircle className="w-3 h-3 mr-1" />
                              Resolve
                            </Button>
                            <Button size="sm" variant="outline" className="border-red-600 text-red-400 hover:bg-red-950/30" onClick={() => onDismiss?.(incident.id)}>
                              <XCircle className="w-3 h-3 mr-1" />
                              Dismiss
                            </Button>
                          </>
                        )}

                        {!isDispatched && incident.status === 'active' && (role === 'admin' || role === 'security' || role === 'officer') && (
                          <div className="flex w-full gap-2">
                            {role === 'security' && (
                              <Button
                                size="sm"
                                className="flex-1 bg-purple-600 hover:bg-purple-700 text-white"
                                onClick={() => onSelfDispatch?.(incident.id)}
                              >
                                <UserCheck className="w-3.5 h-3.5 mr-2" />
                                Acknowledge
                              </Button>
                            )}
                            {(role === 'admin' || role === 'officer') && (
                              <Button
                                size="sm"
                                className="bg-blue-600 hover:bg-blue-700 text-white w-full"
                                onClick={(e: React.MouseEvent) => handleOpenDispatch(incident.id, e)}
                              >
                                <UserPlus className="w-3.5 h-3.5 mr-2" />
                                Dispatch
                              </Button>
                            )}
                          </div>
                        )}
                        {incident.status === 'dispatched' && incident.assigned_security && (
                          <div className="text-xs px-2 py-1 rounded w-full text-center font-medium text-black bg-blue-200 border border-blue-300 dark:text-blue-300 dark:bg-blue-950/50 dark:border-blue-500/30">
                            Officer {incident.assigned_security}
                          </div>
                        )}
                        {incident.status === 'acknowledged' && incident.ack_by && (
                          <div className="text-xs px-2 py-1 rounded w-full text-center font-medium text-black bg-purple-200 border border-purple-300 dark:text-purple-300 dark:bg-purple-950/50 dark:border-purple-500/30">
                            Ack by {incident.ack_by}
                          </div>
                        )}
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </motion.div>
            );
          }))}
      </div>
    </div>
  );
}