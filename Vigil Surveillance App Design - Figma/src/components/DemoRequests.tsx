import { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Button } from './ui/button';
import { Input } from './ui/input';
import { Badge } from './ui/badge';
import { 
  Calendar, 
  Mail, 
  Phone, 
  Building2, 
  User, 
  MessageSquare, 
  CheckCircle, 
  XCircle, 
  Clock,
  Search,
  Filter
} from 'lucide-react';
import { vigilClasses } from './vigil-theme';

export function DemoRequests() {
  const [searchQuery, setSearchQuery] = useState('');
  const [statusFilter, setStatusFilter] = useState<'all' | 'pending' | 'contacted' | 'scheduled' | 'completed' | 'declined'>('all');

  // Mock demo requests data
  const demoRequests = [
    {
      id: 'DEMO-001',
      name: 'John Martinez',
      email: 'j.martinez@citypolice.gov',
      phone: '+1 (555) 123-4567',
      organization: 'City Police Department',
      position: 'Chief of Security',
      requestDate: '2026-01-10',
      preferredDate: '2026-01-15',
      status: 'pending',
      message: 'Interested in AI-powered surveillance for downtown area monitoring. Looking to replace current system.',
      cameraCount: '50-100',
      budget: '$100k-$250k',
    },
    {
      id: 'DEMO-002',
      name: 'Sarah Chen',
      email: 's.chen@metrosecurity.com',
      phone: '+1 (555) 234-5678',
      organization: 'Metro Security Services',
      position: 'Operations Manager',
      requestDate: '2026-01-09',
      preferredDate: '2026-01-14',
      status: 'contacted',
      message: 'Need a comprehensive demo of violence detection capabilities for shopping mall deployment.',
      cameraCount: '20-50',
      budget: '$50k-$100k',
    },
    {
      id: 'DEMO-003',
      name: 'Michael Thompson',
      email: 'm.thompson@highway-patrol.gov',
      phone: '+1 (555) 345-6789',
      organization: 'State Highway Patrol',
      position: 'Technology Director',
      requestDate: '2026-01-08',
      preferredDate: '2026-01-20',
      status: 'scheduled',
      message: 'Looking for crash detection and automatic alert system for highway network.',
      cameraCount: '100+',
      budget: '$250k+',
    },
    {
      id: 'DEMO-004',
      name: 'Emily Rodriguez',
      email: 'e.rodriguez@campus-security.edu',
      phone: '+1 (555) 456-7890',
      organization: 'University Campus Security',
      position: 'Director of Security',
      requestDate: '2026-01-07',
      preferredDate: '2026-01-12',
      status: 'completed',
      message: 'Campus-wide surveillance upgrade with AI detection for student safety.',
      cameraCount: '50-100',
      budget: '$100k-$250k',
    },
    {
      id: 'DEMO-005',
      name: 'David Park',
      email: 'd.park@small-business.com',
      phone: '+1 (555) 567-8901',
      organization: 'Park Retail Group',
      position: 'Owner',
      requestDate: '2026-01-06',
      preferredDate: '2026-01-11',
      status: 'declined',
      message: 'Small retail chain looking for security solution.',
      cameraCount: '<20',
      budget: '<$50k',
    },
  ];

  const stats = [
    { label: 'Total Requests', value: '47', color: 'text-blue-400', status: 'all' },
    { label: 'Pending Review', value: '12', color: 'text-amber-400', status: 'pending' },
    { label: 'Scheduled', value: '8', color: 'text-green-400', status: 'scheduled' },
    { label: 'This Week', value: '5', color: 'text-cyan-400', status: 'all' },
  ];

  const filteredRequests = demoRequests.filter((request) => {
    const matchesSearch = 
      request.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      request.email.toLowerCase().includes(searchQuery.toLowerCase()) ||
      request.organization.toLowerCase().includes(searchQuery.toLowerCase());
    const matchesStatus = statusFilter === 'all' || request.status === statusFilter;
    return matchesSearch && matchesStatus;
  });

  const getStatusConfig = (status: string) => {
    switch (status) {
      case 'pending':
        return {
          color: 'bg-amber-500/20 text-amber-400 border-amber-500/30',
          icon: Clock,
        };
      case 'contacted':
        return {
          color: 'bg-blue-500/20 text-blue-400 border-blue-500/30',
          icon: Mail,
        };
      case 'scheduled':
        return {
          color: 'bg-green-500/20 text-green-400 border-green-500/30',
          icon: Calendar,
        };
      case 'completed':
        return {
          color: 'bg-cyan-500/20 text-cyan-400 border-cyan-500/30',
          icon: CheckCircle,
        };
      case 'declined':
        return {
          color: 'bg-red-500/20 text-red-400 border-red-500/30',
          icon: XCircle,
        };
      default:
        return {
          color: 'bg-gray-500/20 text-gray-400 border-gray-500/30',
          icon: Clock,
        };
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <Calendar className="w-8 h-8 text-blue-600 dark:text-blue-400" />
          <div>
            <h1 className="text-2xl font-bold tracking-tight">Demo Requests</h1>
            <p className={vigilClasses.textMuted}>
              Manage and track demo booking requests from potential clients
            </p>
          </div>
        </div>
        <Button className="bg-gradient-to-r from-cyan-600 to-blue-600 hover:from-cyan-700 hover:to-blue-700 text-white">
          <Calendar className="w-5 h-5 mr-2" />
          Schedule Demo
        </Button>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        {stats.map((stat) => (
          <Card 
            key={stat.label} 
            className={`${vigilClasses.card} cursor-pointer hover:border-cyan-500/30 transition-all`}
            onClick={() => setStatusFilter(stat.status as any)}
          >
            <CardContent className="p-4">
              <p className={`text-xs ${vigilClasses.textMuted} mb-1`}>{stat.label}</p>
              <p className={`text-2xl font-bold ${stat.color}`}>{stat.value}</p>
            </CardContent>
          </Card>
        ))}
      </div>

      {/* Filters */}
      <Card className={vigilClasses.card}>
        <CardContent className="p-4">
          <div className="flex flex-col md:flex-row gap-4">
            {/* Search */}
            <div className="flex-1 relative">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />
              <Input
                placeholder="Search by name, email, or organization..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="pl-10"
              />
            </div>

            {/* Status Filter */}
            <div className="flex gap-2 flex-wrap">
              {['all', 'pending', 'contacted', 'scheduled', 'completed', 'declined'].map((status) => (
                <Button
                  key={status}
                  variant={statusFilter === status ? 'default' : 'outline'}
                  size="sm"
                  onClick={() => setStatusFilter(status as typeof statusFilter)}
                  className="capitalize"
                >
                  {status}
                </Button>
              ))}
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Requests List */}
      <div className="space-y-3">
        {filteredRequests.map((request) => {
          const statusConfig = getStatusConfig(request.status);
          const StatusIcon = statusConfig.icon;

          return (
            <Card key={request.id} className={`${vigilClasses.card} hover:border-cyan-500/30 transition-all`}>
              <CardContent className="p-5">
                <div className="flex items-start justify-between gap-4">
                  {/* Request Info */}
                  <div className="flex-1 space-y-3">
                    {/* Header */}
                    <div className="flex items-center gap-3">
                      <div className="w-10 h-10 rounded-full bg-gradient-to-br from-cyan-600 to-blue-600 flex items-center justify-center text-white font-bold">
                        {request.name.charAt(0)}
                      </div>
                      <div className="flex-1">
                        <div className="flex items-center gap-2">
                          <h3 className={`font-semibold ${vigilClasses.textPrimary}`}>
                            {request.name}
                          </h3>
                          <Badge className={statusConfig.color}>
                            <StatusIcon className="w-3 h-3 mr-1" />
                            {request.status}
                          </Badge>
                        </div>
                        <p className={`text-sm ${vigilClasses.textMuted}`}>
                          {request.position} • {request.organization}
                        </p>
                      </div>
                    </div>

                    {/* Contact Info */}
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                      <div className="flex items-center gap-2">
                        <Mail className="w-4 h-4 text-cyan-400" />
                        <span className={`text-sm ${vigilClasses.textSecondary}`}>{request.email}</span>
                      </div>
                      <div className="flex items-center gap-2">
                        <Phone className="w-4 h-4 text-green-400" />
                        <span className={`text-sm ${vigilClasses.textSecondary}`}>{request.phone}</span>
                      </div>
                      <div className="flex items-center gap-2">
                        <Calendar className="w-4 h-4 text-amber-400" />
                        <span className={`text-sm ${vigilClasses.textSecondary}`}>
                          Preferred: {request.preferredDate}
                        </span>
                      </div>
                    </div>

                    {/* Message */}
                    <div className={`p-3 rounded-lg ${vigilClasses.bgCard} border ${vigilClasses.border}`}>
                      <div className="flex items-start gap-2">
                        <MessageSquare className="w-4 h-4 text-gray-400 mt-0.5 flex-shrink-0" />
                        <p className={`text-sm ${vigilClasses.textSecondary}`}>{request.message}</p>
                      </div>
                    </div>

                    {/* Additional Info */}
                    <div className="flex items-center gap-6">
                      <div className="flex items-center gap-2">
                        <span className={`text-xs ${vigilClasses.textMuted}`}>Cameras:</span>
                        <Badge variant="outline" className="text-xs">
                          {request.cameraCount}
                        </Badge>
                      </div>
                      <div className="flex items-center gap-2">
                        <span className={`text-xs ${vigilClasses.textMuted}`}>Budget:</span>
                        <Badge variant="outline" className="text-xs">
                          {request.budget}
                        </Badge>
                      </div>
                      <div className="flex items-center gap-2">
                        <span className={`text-xs ${vigilClasses.textMuted}`}>Requested:</span>
                        <span className={`text-xs ${vigilClasses.textSecondary}`}>{request.requestDate}</span>
                      </div>
                    </div>
                  </div>

                  {/* Actions */}
                  <div className="flex flex-col gap-2">
                    {request.status === 'pending' && (
                      <>
                        <Button size="sm" className="bg-green-600 hover:bg-green-700 text-white">
                          <CheckCircle className="w-4 h-4 mr-1" />
                          Accept
                        </Button>
                        <Button size="sm" variant="outline" className="border-blue-500/50 text-blue-400 hover:bg-blue-500/10">
                          <Mail className="w-4 h-4 mr-1" />
                          Contact
                        </Button>
                        <Button size="sm" variant="outline" className="border-red-500/50 text-red-400 hover:bg-red-500/10">
                          <XCircle className="w-4 h-4 mr-1" />
                          Decline
                        </Button>
                      </>
                    )}
                    {request.status === 'contacted' && (
                      <>
                        <Button size="sm" className="bg-cyan-600 hover:bg-cyan-700 text-white">
                          <Calendar className="w-4 h-4 mr-1" />
                          Schedule
                        </Button>
                        <Button size="sm" variant="outline" className="border-gray-500/50 text-gray-400 hover:bg-gray-500/10">
                          <MessageSquare className="w-4 h-4 mr-1" />
                          Follow Up
                        </Button>
                      </>
                    )}
                    {request.status === 'scheduled' && (
                      <>
                        <Button size="sm" className="bg-blue-600 hover:bg-blue-700 text-white">
                          <Calendar className="w-4 h-4 mr-1" />
                          View Details
                        </Button>
                        <Button size="sm" variant="outline" className="border-amber-500/50 text-amber-400 hover:bg-amber-500/10">
                          Reschedule
                        </Button>
                      </>
                    )}
                    {request.status === 'completed' && (
                      <Button size="sm" variant="outline" className="border-cyan-500/50 text-cyan-400 hover:bg-cyan-500/10">
                        View Notes
                      </Button>
                    )}
                    {request.status === 'declined' && (
                      <Button size="sm" variant="outline" disabled className="opacity-50">
                        Declined
                      </Button>
                    )}
                  </div>
                </div>
              </CardContent>
            </Card>
          );
        })}
      </div>

      {filteredRequests.length === 0 && (
        <Card className={vigilClasses.card}>
          <CardContent className="p-12 text-center">
            <Calendar className="w-16 h-16 mx-auto text-gray-400 mb-4" />
            <p className={`text-lg ${vigilClasses.textSecondary}`}>No demo requests found</p>
            <p className={vigilClasses.textMuted}>
              Try adjusting your search or filter criteria
            </p>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
