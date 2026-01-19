import { useState } from 'react';
import { vigilClasses } from './vigil-theme';
import { Button } from './ui/button';
import { Card } from './ui/card';
import { Badge } from './ui/badge';
import {
  FileText,
  BarChart3,
  Camera,
  TrendingUp,
  Download,
  Eye,
  Share2,
  Trash2,
  FileBarChart,
  Filter,
} from 'lucide-react';
import { motion, AnimatePresence } from 'motion/react';
import { toast } from 'sonner';
import { ReportGenerationModal } from './ReportGenerationModal';

import { SHARED_REPORTS, Report } from '../data/reportsData';

export function OfficerReports() {
  const [reports, setReports] = useState<Report[]>(SHARED_REPORTS);
  const [reportType, setReportType] = useState('all');
  const [dateFrom, setDateFrom] = useState('');
  const [dateTo, setDateTo] = useState('');
  const [statusFilter, setStatusFilter] = useState('all');

  // Modal State
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [modalDefaultTab, setModalDefaultTab] = useState<'general' | 'incidents' | 'analytics' | 'cameras' | 'performance'>('general');

  const reportTypes = [
    {
      id: 'incidents',
      title: 'Incidents Report',
      description: 'Comprehensive incident logs with AI detection details, timestamps, and resolution status',
      icon: FileText,
      count: reports.filter(r => r.type === 'incidents' || r.type === 'violence' || r.type === 'crash').length,
      color: 'text-red-500 dark:text-red-400',
      bgColor: 'bg-red-500/20',
      modalTab: 'incidents'
    },
    {
      id: 'analytics',
      title: 'Analytics Report',
      description: 'Performance metrics, detection accuracy, response times, and trend analysis',
      icon: BarChart3,
      count: reports.filter(r => r.type === 'analytics').length,
      color: 'text-amber-600 dark:text-amber-400',
      bgColor: 'bg-amber-500/20',
      modalTab: 'analytics'
    },
    {
      id: 'cameras',
      title: 'Camera Status Report',
      description: 'Camera health status, uptime statistics, and maintenance schedules',
      icon: Camera,
      count: reports.filter(r => r.type === 'cameras').length,
      color: 'text-blue-600 dark:text-blue-400',
      bgColor: 'bg-blue-500/20',
      modalTab: 'cameras'
    },
    {
      id: 'performance',
      title: 'Performance Report',
      description: 'Officer response metrics, system performance, and efficiency analysis',
      icon: TrendingUp,
      count: reports.filter(r => r.type === 'performance').length,
      color: 'text-green-600 dark:text-green-400',
      bgColor: 'bg-green-500/20',
      modalTab: 'performance'
    },
  ];

  // Filter logic
  const filteredReports = reports.filter(report => {
    // Type Filter
    if (reportType !== 'all') {
      if (reportType === 'incidents') {
        if (!['incidents', 'violence', 'crash'].includes(report.type)) return false;
      } else {
        if (report.type !== reportType) return false;
      }
    }
    // Status Filter
    if (statusFilter !== 'all' && report.status !== statusFilter) return false;

    // Date Filter (Basic string comparison for demo)
    if (dateFrom && report.generatedDate < dateFrom) return false;
    if (dateTo && report.generatedDate > dateTo) return false;

    return true;
  });

  const handleOpenGenerateModal = (tab: any = 'general') => {
    setModalDefaultTab(tab);
    setIsModalOpen(true);
  };

  const mapIdToTab = (id: string) => {
    if (id === 'incidents') return 'incidents';
    if (id === 'analytics') return 'analytics';
    if (id === 'cameras') return 'cameras';
    return 'general';
  }

  const handleDeleteReport = (id: string) => {
    setReports(reports.filter(r => r.id !== id));
    toast.success("Report deleted");
  };

  const handleDownloadReport = (report: Report) => {
    toast.success(`Downloading ${report.title}...`);
  };

  const handleViewReport = (report: Report) => {
    toast.info(`Opening ${report.title}`);
  };

  const handleShareReport = (report: Report) => {
    toast.success("Report link copied to clipboard");
  };

  const getTypeIcon = (type: string) => {
    switch (type) {
      case 'incidents':
      case 'violence':
      case 'crash':
        return FileText;
      case 'analytics':
        return BarChart3;
      case 'cameras':
        return Camera;
      case 'performance':
        return TrendingUp;
      default:
        return FileText;
    }
  };

  const getTypeBadgeColor = (type: string) => {
    switch (type) {
      case 'incidents':
      case 'violence':
      case 'crash':
        return 'bg-red-500/20 text-red-600 dark:text-red-400 border-red-500/30';
      case 'analytics':
        return 'bg-amber-500/20 text-amber-600 dark:text-amber-400 border-amber-500/30';
      case 'cameras':
        return 'bg-blue-500/20 text-blue-600 dark:text-blue-400 border-blue-500/30';
      case 'performance':
        return 'bg-green-500/20 text-green-600 dark:text-green-400 border-green-500/30';
      default:
        return 'bg-gray-500/20 text-gray-600 dark:text-gray-400 border-gray-500/30';
    }
  };

  const handleReportGenerated = (newReport: Report) => {
    setReports([newReport, ...reports]);
  };

  return (
    <div className="space-y-6">
      <ReportGenerationModal
        isOpen={isModalOpen}
        onClose={() => setIsModalOpen(false)}
        role="officer"
        defaultTab={modalDefaultTab}
        onGenerate={handleReportGenerated}
      />

      {/* Header */}
      <div className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
        <div>
          <h2 className={`text-3xl mb-2 ${vigilClasses.textPrimary}`}>
            Reports
          </h2>
          <p className={vigilClasses.textMuted}>
            Generate and manage your incident and performance reports
          </p>
        </div>
        <div className="flex gap-3">
          <Button
            className="bg-green-600 hover:bg-green-700 text-white px-6"
            onClick={() => toast.info("Exporting all reports...")}
          >
            <Download className="w-4 h-4 mr-2" />
            Export Reports
          </Button>
          <Button
            className="bg-amber-600 hover:bg-amber-700 text-white px-6"
            onClick={() => handleOpenGenerateModal('general')}
          >
            <FileBarChart className="w-4 h-4 mr-2" />
            Generate Report
          </Button>
        </div>
      </div>

      {/* Filters Section */}
      <Card className={`p-6 ${vigilClasses.card}`}>
        <div className="flex flex-col gap-4">
          <div className="flex items-center gap-2 mb-2">
            <Filter className="w-5 h-5 text-amber-600 dark:text-amber-400" />
            <h3 className={`text-lg ${vigilClasses.textPrimary}`}>Filters</h3>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-5 gap-4">
            {/* Report Type */}
            <div>
              <label className={`text-sm mb-2 block ${vigilClasses.textMuted}`}>
                Report Type
              </label>
              <select
                value={reportType}
                onChange={(e) => setReportType(e.target.value)}
                className={`w-full px-3 py-2 rounded-lg border ${vigilClasses.border} ${vigilClasses.bgCard} ${vigilClasses.textPrimary} focus:outline-none focus:ring-2 focus:ring-amber-500`}
              >
                <option value="all">All Reports</option>
                <option value="incidents">Incidents</option>
                <option value="analytics">Analytics</option>
                <option value="cameras">Cameras</option>
                <option value="performance">Performance</option>
              </select>
            </div>

            {/* Date From */}
            <div>
              <label className={`text-sm mb-2 block ${vigilClasses.textMuted}`}>
                Date From
              </label>
              <input
                type="date"
                value={dateFrom}
                onChange={(e) => setDateFrom(e.target.value)}
                className={`w-full px-3 py-2 rounded-lg border ${vigilClasses.border} ${vigilClasses.bgCard} ${vigilClasses.textPrimary} focus:outline-none focus:ring-2 focus:ring-amber-500`}
              />
            </div>

            {/* Date To */}
            <div>
              <label className={`text-sm mb-2 block ${vigilClasses.textMuted}`}>
                Date To
              </label>
              <input
                type="date"
                value={dateTo}
                onChange={(e) => setDateTo(e.target.value)}
                className={`w-full px-3 py-2 rounded-lg border ${vigilClasses.border} ${vigilClasses.bgCard} ${vigilClasses.textPrimary} focus:outline-none focus:ring-2 focus:ring-amber-500`}
              />
            </div>

            {/* Status */}
            <div>
              <label className={`text-sm mb-2 block ${vigilClasses.textMuted}`}>
                Status
              </label>
              <select
                value={statusFilter}
                onChange={(e) => setStatusFilter(e.target.value)}
                className={`w-full px-3 py-2 rounded-lg border ${vigilClasses.border} ${vigilClasses.bgCard} ${vigilClasses.textPrimary} focus:outline-none focus:ring-2 focus:ring-amber-500`}
              >
                <option value="all">All</option>
                <option value="completed">Completed</option>
                <option value="processing">Processing</option>
              </select>
            </div>

            {/* Apply Filters Button */}
            <div className="flex items-end">
              <Button
                className="w-full bg-amber-600 hover:bg-amber-700 text-white"
                onClick={() => toast.success("Filters applied")}
              >
                Apply Filters
              </Button>
            </div>
          </div>
        </div>
      </Card>

      {/* Report Type Cards */}
      <div>
        <h3 className={`text-xl mb-4 ${vigilClasses.textPrimary}`}>
          Report Types
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          {reportTypes.map((rt, index) => {
            const Icon = rt.icon;
            return (
              <motion.div
                key={rt.id}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: index * 0.1 }}
                onClick={() => setReportType(rt.id)}
              >
                <Card
                  className={`${vigilClasses.card} p-5 hover:border-amber-500/50 dark:hover:border-amber-500/50 transition-all cursor-pointer group h-full flex flex-col ${reportType === rt.id ? 'ring-2 ring-amber-500' : ''}`}
                >
                  <div className="flex items-start justify-between mb-4">
                    <div
                      className={`w-12 h-12 rounded-full ${rt.bgColor} flex items-center justify-center group-hover:scale-110 transition-transform`}
                    >
                      <Icon className={`w-6 h-6 ${rt.color}`} />
                    </div>
                    <Badge className={getTypeBadgeColor(rt.id)}>
                      {rt.count}
                    </Badge>
                  </div>
                  <h4 className={`mb-2 ${vigilClasses.textPrimary}`}>
                    {rt.title}
                  </h4>
                  <p className={`text-sm ${vigilClasses.textMuted} mb-4 flex-1`}>
                    {rt.description}
                  </p>
                  <Button
                    size="sm"
                    className="w-full bg-amber-600 hover:bg-amber-700 text-white"
                    onClick={(e: React.MouseEvent) => {
                      e.stopPropagation();
                      setReportType(rt.id);
                      handleOpenGenerateModal(mapIdToTab(rt.id));
                    }}
                  >
                    Generate Report
                  </Button>
                </Card>
              </motion.div>
            );
          })}
        </div>
      </div>

      {/* Reports Table */}
      <div>
        <h3 className={`text-xl mb-4 ${vigilClasses.textPrimary}`}>
          Recent Reports
        </h3>
        <Card className={`${vigilClasses.card} overflow-hidden`}>
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className={`border-b ${vigilClasses.border}`}>
                  <th className={`text-left p-4 ${vigilClasses.textMuted}`}>
                    Report
                  </th>
                  <th className={`text-left p-4 ${vigilClasses.textMuted}`}>
                    Type
                  </th>
                  <th className={`text-left p-4 ${vigilClasses.textMuted}`}>
                    Status
                  </th>
                  <th className={`text-left p-4 ${vigilClasses.textMuted}`}>
                    Generated
                  </th>
                  <th className={`text-left p-4 ${vigilClasses.textMuted}`}>
                    Size
                  </th>
                  <th className={`text-left p-4 ${vigilClasses.textMuted}`}>
                    Actions
                  </th>
                </tr>
              </thead>
              <tbody>
                <AnimatePresence>
                  {filteredReports.length === 0 ? (
                    <tr>
                      <td colSpan={6} className="p-8 text-center">
                        <FileText className={`w-12 h-12 mx-auto mb-3 ${vigilClasses.textMuted}`} />
                        <p className={`text-lg ${vigilClasses.textPrimary}`}>
                          No reports found
                        </p>
                        <p className={`text-sm ${vigilClasses.textMuted}`}>
                          Generate your first report to get started
                        </p>
                      </td>
                    </tr>
                  ) : (
                    filteredReports.map((report, index) => {
                      const TypeIcon = getTypeIcon(report.type);
                      return (
                        <motion.tr
                          key={report.id}
                          initial={{ opacity: 0, x: -20 }}
                          animate={{ opacity: 1, x: 0 }}
                          exit={{ opacity: 0, height: 0 }}
                          transition={{ delay: index * 0.05 }}
                          className={`border-b ${vigilClasses.border} ${vigilClasses.bgCardHover} transition-colors`}
                        >
                          {/* Report */}
                          <td className="p-4">
                            <div className="flex items-center gap-3">
                              <div className={`w-10 h-10 rounded ${vigilClasses.bgCard} flex items-center justify-center`}>
                                <TypeIcon className="w-5 h-5 text-amber-600 dark:text-amber-400" />
                              </div>
                              <div>
                                <p className={`${vigilClasses.textPrimary}`}>
                                  {report.title}
                                </p>
                                <p className={`text-sm ${vigilClasses.textMuted}`}>
                                  {report.id} • {report.format}
                                </p>
                              </div>
                            </div>
                          </td>

                          {/* Type */}
                          <td className="p-4">
                            <Badge className={getTypeBadgeColor(report.type)}>
                              {report.type}
                            </Badge>
                          </td>

                          {/* Status */}
                          <td className="p-4">
                            {report.status === 'completed' ? (
                              <Badge className="bg-green-500/20 text-green-600 dark:text-green-400 border-green-500/30">
                                Completed
                              </Badge>
                            ) : (
                              <Badge className="bg-amber-500/20 text-amber-600 dark:text-amber-400 border-amber-500/30">
                                <div className="w-2 h-2 bg-amber-600 dark:bg-amber-400 rounded-full animate-pulse mr-1.5" />
                                Processing
                              </Badge>
                            )}
                          </td>

                          {/* Generated */}
                          <td className="p-4">
                            <p className={vigilClasses.textPrimary}>
                              {report.generatedDate}
                            </p>
                          </td>

                          {/* Size */}
                          <td className="p-4">
                            <p className={vigilClasses.textPrimary}>{report.size}</p>
                          </td>

                          {/* Actions */}
                          <td className="p-4">
                            <div className="flex items-center gap-2">
                              {report.status === 'completed' && (
                                <>
                                  <Button
                                    size="sm"
                                    variant="outline"
                                    className="border-amber-600 text-amber-600 dark:text-amber-400 hover:bg-amber-50 dark:hover:bg-amber-950/30"
                                    onClick={() => handleDownloadReport(report)}
                                  >
                                    <Download className="w-3 h-3 mr-1" />
                                    Download
                                  </Button>
                                  <Button
                                    size="sm"
                                    variant="outline"
                                    className={`${vigilClasses.border} hover:${vigilClasses.bgCard}`}
                                    onClick={() => handleViewReport(report)}
                                  >
                                    <Eye className="w-3 h-3 mr-1" />
                                    View
                                  </Button>
                                  <Button
                                    size="sm"
                                    variant="outline"
                                    className={`${vigilClasses.border} hover:${vigilClasses.bgCard}`}
                                    onClick={() => handleShareReport(report)}
                                  >
                                    <Share2 className="w-3 h-3" />
                                  </Button>
                                  <Button
                                    size="sm"
                                    variant="outline"
                                    className="border-red-600 text-red-600 dark:text-red-400 hover:bg-red-50 dark:hover:bg-red-950/30"
                                    onClick={() => handleDeleteReport(report.id)}
                                  >
                                    <Trash2 className="w-3 h-3" />
                                  </Button>
                                </>
                              )}
                              {report.status === 'processing' && (
                                <span className={`text-sm ${vigilClasses.textMuted}`}>
                                  Generating...
                                </span>
                              )}
                            </div>
                          </td>
                        </motion.tr>
                      );
                    })
                  )}
                </AnimatePresence>
              </tbody>
            </table>
          </div>
        </Card>
      </div>
    </div>
  );
}