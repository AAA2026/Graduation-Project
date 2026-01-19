import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import {
    Shield,
    Clock,
    MapPin,
    CheckCircle,
    XCircle,
    AlertTriangle,
    Radio,
    User,
    Navigation,
    Siren,
    ChevronRight,
    AlertCircle
} from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Button } from './ui/button';
import { Badge } from './ui/badge';
import { vigilClasses } from './vigil-theme';
import { useTheme } from './ThemeProvider';
import type { Incident } from '../hooks/useRealtimeIncidents';

interface DispatchesViewProps {
    role: 'admin' | 'officer' | 'security';
    incidents: Incident[];
    resolvedIncidents?: Incident[];
    title?: string;
    onResolve?: (id: string, type: 'resolved' | 'not_resolved') => void;
    onDismiss?: (id: string) => void;
    onDispatch?: (id: string, securityId: string) => void;
    onConfirm?: (id: string) => void;
    onReject?: (id: string) => void;
    onDismissAll?: () => void;
    onIncidentClick?: (id: string) => void;
}

export function DispatchesView({
    role,
    incidents,
    resolvedIncidents = [],
    title = "Active Missions",
    onResolve,
    onDismiss,
    onConfirm,
    onReject,
    onDismissAll,
    onIncidentClick
}: DispatchesViewProps) {
    const { theme } = useTheme();

    // Sort: High priority & Newest first
    const sortIncidents = (list: Incident[]) => [...list].sort((a, b) => {
        if (a.severity === 'critical' && b.severity !== 'critical') return -1;
        if (b.severity === 'critical' && a.severity !== 'critical') return 1;
        return new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime();
    });

    const sortedActive = sortIncidents(incidents);
    const sortedResolved = sortIncidents(resolvedIncidents);

    const showSplitView = resolvedIncidents.length > 0;

    return (
        <div className="space-y-8">
            {/* Active Section */}
            <section>
                <div className="flex items-center justify-between mb-6">
                    <div className="flex items-center gap-3">
                        <div className="p-2 bg-blue-500/20 rounded-lg border border-blue-500/30">
                            <Shield className="w-6 h-6 text-blue-400" />
                        </div>
                        <div>
                            <h1 className="text-2xl font-bold tracking-tight">{title}</h1>
                            {/* If split view, explicitly say Active. If not, maybe just title. Actually keeping title generic is fine, but user wants split. */}
                            {showSplitView && <p className="text-sm font-semibold text-blue-400 mt-1 uppercase tracking-wider">Active Missions</p>}
                        </div>
                    </div>
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
                </div>

                <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-6">
                    <AnimatePresence mode="popLayout">
                        {sortedActive.map((incident) => (
                            <MissionCard
                                key={incident.id}
                                incident={incident}
                                role={role}
                                onConfirm={onConfirm}
                                onReject={onReject}
                                onResolve={onResolve}
                                onClick={() => onIncidentClick?.(incident.id)}
                            />
                        ))}
                    </AnimatePresence>

                    {sortedActive.length === 0 && (
                        <div className="col-span-full flex flex-col items-center justify-center p-12 text-center border-2 border-dashed border-gray-200 dark:border-gray-800 rounded-xl bg-white/50 dark:bg-black/20 backdrop-blur-sm">
                            <Shield className="w-12 h-12 text-gray-300 dark:text-gray-700 mb-4" />
                            <h3 className="text-lg font-medium text-gray-500">No active dispatches</h3>
                            <p className="text-gray-400">You have no active missions at this time.</p>
                        </div>
                    )}
                </div>
            </section>

            {/* Resolved Section (Only show if we have resolved incidents passed) */}
            {sortedResolved.length > 0 && (
                <section className="pt-8 border-t border-gray-200 dark:border-gray-800">
                    <div className="flex items-center gap-3 mb-6 opacity-80">
                        <div className="p-2 bg-green-500/10 rounded-lg border border-green-500/20">
                            <CheckCircle className="w-5 h-5 text-green-500" />
                        </div>
                        <h2 className="text-xl font-bold tracking-tight text-gray-700 dark:text-gray-300">Resolved Missions</h2>
                    </div>

                    <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-6 opacity-75 hover:opacity-100 transition-opacity">
                        {sortedResolved.map((incident) => (
                            <MissionCard
                                key={incident.id}
                                incident={incident}
                                role={role}
                                onConfirm={onConfirm} // Actions hidden for resolved usually, but pass just in case layout needs them
                                onReject={onReject}
                                onResolve={onResolve}
                                isDisabled
                                onClick={() => onIncidentClick?.(incident.id)}
                            />
                        ))}
                    </div>
                </section>
            )}
        </div>
    );
}

// Add isDisabled prop to MissionCard
function MissionCard({ incident, role, onConfirm, onReject, onResolve, onClick, isDisabled = false }: {
    incident: Incident,
    role: string,
    onConfirm?: any,
    onReject?: any,
    onResolve?: any,
    onClick?: () => void,
    isDisabled?: boolean
}) {
    // Use a bespoke distinct look
    const isCritical = incident.severity === 'critical';
    const isViolence = incident.type === 'violence';
    const isUnresolved = incident.resolution_type === 'not_resolved'; // Check for Unresolved tag

    return (
        <motion.div
            layout
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.95 }}
            transition={{ duration: 0.2 }}
        >
            <Card
                className={`overflow-hidden border-t-4 transition-all hover:shadow-lg cursor-pointer ${isDisabled ? 'grayscale-[0.5] bg-gray-50/50 dark:bg-gray-900/20 border-t-gray-400' :
                    isCritical ? 'border-t-red-500 shadow-red-500/10' : 'border-t-blue-500 shadow-blue-500/10'
                    }`}
                onClick={onClick}
            >
                <div className="relative">
                    {/* Status Banner */}
                    <div className={`px-4 py-2 flex items-center justify-between ${isDisabled ? 'bg-gray-100 dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700' :
                        incident.status === 'acknowledged'
                            ? 'bg-purple-500/10 border-b border-purple-500/20'
                            : 'bg-blue-500/10 border-b border-blue-500/20'
                        }`}>
                        <div className="flex items-center gap-2">
                            {isDisabled ? (
                                <Badge variant="outline" className="text-gray-500 border-gray-400 bg-gray-100 dark:bg-gray-800">
                                    RESOLVED
                                </Badge>
                            ) : (
                                <Badge variant="outline" className={`font-mono uppercase tracking-wider ${incident.status === 'acknowledged' ? 'text-purple-400 border-purple-500/50' : 'text-blue-400 border-blue-500/50'
                                    }`}>
                                    {incident.status === 'acknowledged' ? 'ON SCENE' : 'EN ROUTE'}
                                </Badge>
                            )}

                            {isCritical && !isDisabled && <Badge variant="destructive" className="animate-pulse">CRITICAL</Badge>}
                            {isUnresolved && !isDisabled && <Badge className="bg-orange-500 hover:bg-orange-600 text-white border-orange-600">UNRESOLVED</Badge>}
                        </div>
                        <div className="flex items-center gap-1 text-xs font-mono opacity-80">
                            <Clock className="w-3 h-3" />
                            <TimeCounter timestamp={incident.timestamp} />
                        </div>
                    </div>

                    <CardContent className="p-5">
                        <div className="flex justify-between items-start mb-4">
                            <div>
                                <h3 className="text-lg font-bold flex items-center gap-2">
                                    {isViolence ? (
                                        <span className="text-red-500 flex items-center gap-1"><Siren className="w-5 h-5" /> Violence Detected</span>
                                    ) : (
                                        <span className="text-orange-500 flex items-center gap-1"><AlertTriangle className="w-5 h-5" /> Car Crash</span>
                                    )}
                                </h3>
                                <p className="text-sm text-muted-foreground mt-1 flex items-center gap-1">
                                    <MapPin className="w-3 h-3" /> {incident.location} • {incident.cameraId}
                                </p>
                            </div>
                            <div className="text-right">
                                <div className="text-2xl font-black text-gray-200">{incident.confidence}%</div>
                                <div className="text-[10px] uppercase text-muted-foreground tracking-widest">Confidence</div>
                            </div>
                        </div>

                        {/* Officer Assignment Info (Visible to All) */}
                        <div className="mb-6 p-3 bg-secondary/30 rounded-lg flex items-center gap-3">
                            <div className="w-10 h-10 rounded-full bg-slate-700 flex items-center justify-center border border-slate-600">
                                <User className="w-5 h-5 text-slate-300" />
                            </div>
                            <div>
                                <p className="text-xs text-muted-foreground uppercase tracking-wider">Assigned Unit</p>
                                <p className="font-semibold text-sm">
                                    {incident.assigned_security || incident.dispatchedOfficerName || "Unassigned"}
                                </p>
                            </div>
                            {incident.status === 'dispatched' && (
                                <Radio className="w-4 h-4 text-blue-500 ml-auto animate-pulse" />
                            )}
                            {incident.status === 'acknowledged' && (
                                <Navigation className="w-4 h-4 text-purple-500 ml-auto" />
                            )}
                        </div>

                        {/* Actions */}
                        <div className="flex gap-3">
                            {role === 'security' && (
                                <>
                                    {incident.status === 'resolved' ? (
                                        <Button
                                            disabled
                                            className="w-full bg-green-500/10 border border-green-500/20 text-green-500 cursor-default hover:bg-green-500/10"
                                        >
                                            <CheckCircle className="w-4 h-4 mr-2" />
                                            Incident Resolved
                                        </Button>
                                    ) : incident.status === 'confirmed' ? (
                                        <>
                                            <Button
                                                className="flex-1 bg-blue-600 hover:bg-blue-700 text-white shadow-lg shadow-blue-900/20"
                                                onClick={(e: React.MouseEvent) => {
                                                    e.stopPropagation();
                                                    // Resolve as 'completed'
                                                    if (onResolve) onResolve(incident.id, 'resolved');
                                                }}
                                            >
                                                <CheckCircle className="w-4 h-4 mr-2" />
                                                Resolve
                                            </Button>
                                            <Button
                                                variant="outline"
                                                className="flex-1 border-orange-500/50 hover:bg-orange-500/10 text-orange-500"
                                                onClick={(e: React.MouseEvent) => {
                                                    e.stopPropagation();
                                                    // Resolve as 'not_resolved' or leave active?
                                                    // Request says: "state if its resolved or not"
                                                    if (onResolve) onResolve(incident.id, 'not_resolved');
                                                }}
                                            >
                                                <AlertCircle className="w-4 h-4 mr-2" />
                                                Unresolved
                                            </Button>
                                        </>
                                    ) : (
                                        <>
                                            <Button
                                                className="flex-1 bg-green-600 hover:bg-green-700 text-white shadow-lg shadow-green-900/20"
                                                onClick={(e: React.MouseEvent) => {
                                                    e.stopPropagation();
                                                    if (onConfirm) onConfirm(incident.id);
                                                    else if (onResolve) onResolve(incident.id, 'resolved');
                                                }}
                                            >
                                                <CheckCircle className="w-4 h-4 mr-2" />
                                                Verified
                                            </Button>
                                            <Button
                                                variant="outline"
                                                className="flex-1 border-red-500/50 hover:bg-red-500/10 text-red-500"
                                                onClick={(e: React.MouseEvent) => {
                                                    e.stopPropagation();
                                                    if (onReject) onReject(incident.id);
                                                    else if (onResolve) onResolve(incident.id, 'not_resolved');
                                                }}
                                            >
                                                <XCircle className="w-4 h-4 mr-2" />
                                                False Alarm
                                            </Button>
                                        </>
                                    )}
                                </>
                            )}
                            {role !== 'security' && (
                                <Button
                                    variant="secondary"
                                    className="w-full"
                                    size="sm"
                                    onClick={(e: React.MouseEvent) => {
                                        e.stopPropagation();
                                        onClick?.();
                                    }}
                                >
                                    View Live Feed <ChevronRight className="w-4 h-4 ml-1" />
                                </Button>
                            )}
                        </div>
                    </CardContent>
                </div>
            </Card>
        </motion.div>
    );
}

function TimeCounter({ timestamp }: { timestamp: string }) {
    const [elapsed, setElapsed] = useState("");

    useEffect(() => {
        const update = () => {
            const diff = Math.floor((Date.now() - new Date(timestamp).getTime()) / 1000);
            if (diff < 60) setElapsed(`${diff}s`);
            else {
                const m = Math.floor(diff / 60);
                setElapsed(`${m}m ${diff % 60}s`);
            }
        };
        update();
        const interval = setInterval(update, 1000);
        return () => clearInterval(interval);
    }, [timestamp]);

    return <span>{elapsed}</span>;
}
