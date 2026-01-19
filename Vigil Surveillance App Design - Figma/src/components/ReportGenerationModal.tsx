import { useState, useEffect } from 'react';
import {
    Dialog,
    DialogContent,
    DialogHeader,
    DialogTitle,
    DialogFooter,
    DialogDescription
} from './ui/dialog';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './ui/tabs';
import { Button } from './ui/button';
import { Label } from './ui/label';
import { Input } from './ui/input';
import {
    Select,
    SelectContent,
    SelectItem,
    SelectTrigger,
    SelectValue
} from './ui/select';
import { Checkbox } from './ui/checkbox';
import {
    FileText,
    BarChart3,
    Camera,
    TrendingUp,
    Calendar as CalendarIcon,
    Download,
    Loader2,
    FileSpreadsheet
} from 'lucide-react';
import { toast } from 'sonner';

interface ReportGenerationModalProps {
    isOpen: boolean;
    onClose: () => void;
    role: 'admin' | 'officer';
    defaultTab?: 'general' | 'incidents' | 'analytics' | 'cameras' | 'performance';
    onGenerate: (report: any) => void;
}

export function ReportGenerationModal({
    isOpen,
    onClose,
    role,
    defaultTab = 'general',
    onGenerate
}: ReportGenerationModalProps) {
    const [activeTab, setActiveTab] = useState(defaultTab);
    const [isGenerating, setIsGenerating] = useState(false);

    // Form State
    const [reportName, setReportName] = useState('');
    const [dateRange, setDateRange] = useState('7days');
    const [format, setFormat] = useState('pdf');
    const [includeCharts, setIncludeCharts] = useState(true);
    const [includeRawData, setIncludeRawData] = useState(false);

    // Reset tab when modal opens with a new default
    useEffect(() => {
        if (isOpen) {
            setActiveTab(defaultTab);
            // Auto-generate a name based on tab
            const dateStr = new Date().toISOString().slice(0, 10);
            const typeStr = defaultTab.charAt(0).toUpperCase() + defaultTab.slice(1);
            setReportName(`${typeStr}_Report_${dateStr}`);
        }
    }, [isOpen, defaultTab]);

    const handleGenerate = () => {
        if (!reportName.trim()) {
            toast.error("Please enter a report name");
            return;
        }

        setIsGenerating(true);
        // Simulate generation delay
        setTimeout(() => {
            setIsGenerating(false);

            // Create new report object
            const newReport = {
                id: `RPT-${new Date().getFullYear()}-${Math.floor(Math.random() * 1000).toString().padStart(3, '0')}`,
                title: reportName,
                type: activeTab === 'general' ? 'custom' : activeTab,
                status: 'completed',
                generatedDate: new Date().toISOString().replace('T', ' ').slice(0, 16),
                generatedBy: role === 'admin' ? 'Admin' : 'Officer',
                size: (Math.random() * 5 + 1).toFixed(1) + ' MB',
                format: format.toUpperCase(),
                incidentsCount: Math.floor(Math.random() * 100),
                camerasCount: Math.floor(Math.random() * 50) + 10
            };

            onGenerate(newReport);
            onClose();
            toast.success(`${format.toUpperCase()} Report "${reportName}" generated successfully`);
        }, 2000);
    };

    return (
        <Dialog open={isOpen} onOpenChange={(open: boolean) => !open && onClose()}>
            <DialogContent className="sm:max-w-[380px] bg-card text-card-foreground border-border">
                <DialogHeader>
                    <DialogTitle className="flex items-center gap-2 text-xl">
                        <FileText className="w-5 h-5 text-amber-500" />
                        Generate New Report
                    </DialogTitle>
                    <DialogDescription>
                        Configure parameters and generate comprehensive system reports.
                    </DialogDescription>
                </DialogHeader>

                <div className="py-4">
                    <Tabs defaultValue={activeTab} value={activeTab} onValueChange={(val: string) => setActiveTab(val as any)} className="w-full">
                        <TabsList className="grid grid-cols-4 mb-2">
                            <TabsTrigger value="general">General</TabsTrigger>
                            <TabsTrigger value="incidents">Incidents</TabsTrigger>
                            <TabsTrigger value="analytics">Analytics</TabsTrigger>
                            <TabsTrigger value="cameras">Cameras</TabsTrigger>
                        </TabsList>

                        {/* Common Fields Area */}
                        <div className="grid gap-3 py-2">
                            <div className="grid gap-2">
                                <Label htmlFor="name">Report Name</Label>
                                <Input
                                    id="name"
                                    value={reportName}
                                    onChange={(e) => setReportName(e.target.value)}
                                    placeholder="e.g. Weekly Incident Summary"
                                />
                            </div>

                            <div className="grid grid-cols-2 gap-4">
                                <div className="grid gap-2">
                                    <Label>Date Range</Label>
                                    <Select value={dateRange} onValueChange={setDateRange}>
                                        <SelectTrigger>
                                            <SelectValue placeholder="Select range" />
                                        </SelectTrigger>
                                        <SelectContent>
                                            <SelectItem value="today">Today</SelectItem>
                                            <SelectItem value="yesterday">Yesterday</SelectItem>
                                            <SelectItem value="7days">Last 7 Days</SelectItem>
                                            <SelectItem value="30days">Last 30 Days</SelectItem>
                                            <SelectItem value="this_month">This Month</SelectItem>
                                            <SelectItem value="last_month">Last Month</SelectItem>
                                            <SelectItem value="custom">Custom Range</SelectItem>
                                        </SelectContent>
                                    </Select>
                                </div>

                                <div className="grid gap-2">
                                    <Label>Format</Label>
                                    <div className="flex bg-muted p-1 rounded-md">
                                        <button
                                            className={`flex-1 flex items-center justify-center text-sm py-1.5 rounded-sm transition-all ${format === 'pdf' ? 'bg-background shadow text-foreground' : 'text-muted-foreground hover:bg-background/50'}`}
                                            onClick={() => setFormat('pdf')}
                                        >
                                            <FileText className="w-4 h-4 mr-1.5" /> PDF
                                        </button>
                                        <button
                                            className={`flex-1 flex items-center justify-center text-sm py-1.5 rounded-sm transition-all ${format === 'csv' ? 'bg-background shadow text-foreground' : 'text-muted-foreground hover:bg-background/50'}`}
                                            onClick={() => setFormat('csv')}
                                        >
                                            <FileSpreadsheet className="w-4 h-4 mr-1.5" /> CSV
                                        </button>
                                        <button
                                            className={`flex-1 flex items-center justify-center text-sm py-1.5 rounded-sm transition-all ${format === 'excel' ? 'bg-background shadow text-foreground' : 'text-muted-foreground hover:bg-background/50'}`}
                                            onClick={() => setFormat('excel')}
                                        >
                                            <FileSpreadsheet className="w-4 h-4 mr-1.5" /> XLS
                                        </button>
                                    </div>
                                </div>
                            </div>
                        </div>

                        {/* Tab Specific Content */}
                        <div className="mt-4 bg-muted/30 p-4 rounded-lg border border-border/50">
                            <TabsContent value="general" className="mt-0">
                                <div className="space-y-3">
                                    <h4 className="text-sm font-medium flex items-center gap-2">
                                        <FileText className="w-4 h-4 text-blue-500" />
                                        General Report Summary
                                    </h4>
                                    <p className="text-xs text-muted-foreground">
                                        Aggregated overview of system activity, including high-level incident counts, camera status summaries, and user activity logs. Best for daily briefings.
                                    </p>
                                </div>
                            </TabsContent>

                            <TabsContent value="incidents" className="mt-0">
                                <div className="space-y-3">
                                    <h4 className="text-sm font-medium flex items-center gap-2">
                                        <TrendingUp className="w-4 h-4 text-red-500" />
                                        Incident Details
                                    </h4>
                                    <div className="grid gap-2">
                                        <div className="flex items-center space-x-2">
                                            <Checkbox id="inc_details" defaultChecked />
                                            <Label htmlFor="inc_details" className="text-sm font-normal">Include AI Confidence Scores</Label>
                                        </div>
                                        <div className="flex items-center space-x-2">
                                            <Checkbox id="inc_snaps" defaultChecked />
                                            <Label htmlFor="inc_snaps" className="text-sm font-normal">Include Incident Thumbnails</Label>
                                        </div>
                                    </div>
                                </div>
                            </TabsContent>

                            <TabsContent value="analytics" className="mt-0">
                                <div className="space-y-3">
                                    <h4 className="text-sm font-medium flex items-center gap-2">
                                        <BarChart3 className="w-4 h-4 text-amber-500" />
                                        Performance Analytics
                                    </h4>
                                    <div className="grid gap-2">
                                        <div className="flex items-center space-x-2">
                                            <Checkbox
                                                id="include_charts"
                                                checked={includeCharts}
                                                onCheckedChange={(c: boolean) => setIncludeCharts(!!c)}
                                            />
                                            <Label htmlFor="include_charts" className="text-sm font-normal">Include Visual Charts</Label>
                                        </div>
                                        <div className="flex items-center space-x-2">
                                            <Checkbox
                                                id="raw_data"
                                                checked={includeRawData}
                                                onCheckedChange={(c: boolean) => setIncludeRawData(!!c)}
                                            />
                                            <Label htmlFor="raw_data" className="text-sm font-normal">Include Raw Data Tables</Label>
                                        </div>
                                    </div>
                                </div>
                            </TabsContent>

                            <TabsContent value="cameras" className="mt-0">
                                <div className="space-y-3">
                                    <h4 className="text-sm font-medium flex items-center gap-2">
                                        <Camera className="w-4 h-4 text-cyan-500" />
                                        Camera Health & Uptime
                                    </h4>
                                    <p className="text-xs text-muted-foreground">
                                        Detailed status report for all connected feeds, including downtime logs, bandwidth usage, and maintenance alerts.
                                    </p>
                                </div>
                            </TabsContent>
                        </div>

                    </Tabs>
                </div>

                <DialogFooter className="sm:justify-between gap-2">
                    {/* Schedule Button (Mock) */}
                    <Button variant="ghost" size="sm" className="text-muted-foreground hover:text-foreground" onClick={() => toast.info("Schedule feature coming soon")}>
                        <CalendarIcon className="w-4 h-4 mr-2" />
                        Schedule Report
                    </Button>

                    <div className="flex gap-2">
                        <Button variant="outline" onClick={onClose} disabled={isGenerating}>Cancel</Button>
                        <Button onClick={handleGenerate} disabled={isGenerating} className="bg-amber-600 hover:bg-amber-700 text-white min-w-[120px]">
                            {isGenerating ? (
                                <>
                                    <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                                    Generating...
                                </>
                            ) : (
                                <>
                                    <Download className="w-4 h-4 mr-2" />
                                    Generate
                                </>
                            )}
                        </Button>
                    </div>
                </DialogFooter>
            </DialogContent>
        </Dialog>
    );
}
