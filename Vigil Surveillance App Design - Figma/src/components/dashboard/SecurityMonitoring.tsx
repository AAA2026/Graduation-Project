import { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '../ui/card';
import { Badge } from '../ui/badge';
import { Button } from '../ui/button';
import { Avatar, AvatarFallback, AvatarImage } from '../ui/avatar';
import { Input } from '../ui/input';
import { Label } from '../ui/label';
import { Separator } from '../ui/separator';
import { vigilClasses } from '../vigil-theme';
import { 
  Eye, Users, Clock, CheckCircle, AlertCircle, 
  Search, Filter, MessageSquare, Phone, MapPin,
  Shield, UserCheck, Activity, Download
} from 'lucide-react';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "../ui/select";

interface SecurityPersonnel {
  id: string;
  name: string;
  badgeNumber: string;
  avatar: string;
  status: 'active' | 'responding' | 'available' | 'offline' | 'on-break';
  location: string;
  currentIncident?: {
    id: string;
    title: string;
    type: string;
    responseTime: number; // minutes
  };
  lastVerified: string;
  assignedOfficer: string;
  performance: {
    avgResponse: number;
    incidentsHandled: number;
    successRate: number;
  };
}

export function SecurityMonitoring() {
  const [selectedPersonnel, setSelectedPersonnel] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [statusFilter, setStatusFilter] = useState<string>('all');

  // Mock data - replace with API call
  const securityPersonnel: SecurityPersonnel[] = [
    {
      id: 'sec-001',
      name: 'Alex Chen',
      badgeNumber: 'SG-042',
      avatar: '/avatars/alex.jpg',
      status: 'active',
      location: 'Downtown Sector',
      currentIncident: {
        id: 'inc-789',
        title: 'Mall Disturbance',
        type: 'Fight',
        responseTime: 4
      },
      lastVerified: '10:30 AM',
      assignedOfficer: 'You',
      performance: {
        avgResponse: 5.2,
        incidentsHandled: 42,
        successRate: 94
      }
    },
    {
      id: 'sec-002',
      name: 'Maria Garcia',
      badgeNumber: 'SG-043',
      avatar: '/avatars/maria.jpg',
      status: 'responding',
      location: 'Financial District',
      currentIncident: {
        id: 'inc-790',
        title: 'Bank Alarm',
        type: 'Security Breach',
        responseTime: 8
      },
      lastVerified: '11:15 AM',
      assignedOfficer: 'You',
      performance: {
        avgResponse: 6.8,
        incidentsHandled: 38,
        successRate: 89
      }
    },
    {
      id: 'sec-003',
      name: 'David Park',
      badgeNumber: 'SG-044',
      avatar: '/avatars/david.jpg',
      status: 'available',
      location: 'Residential Zone',
      lastVerified: '9:45 AM',
      assignedOfficer: 'You',
      performance: {
        avgResponse: 4.5,
        incidentsHandled: 56,
        successRate: 97
      }
    },
    {
      id: 'sec-004',
      name: 'Sarah Johnson',
      badgeNumber: 'SG-045',
      avatar: '/avatars/sarah.jpg',
      status: 'on-break',
      location: 'Commercial Area',
      lastVerified: '8:30 AM',
      assignedOfficer: 'Officer Smith',
      performance: {
        avgResponse: 7.2,
        incidentsHandled: 31,
        successRate: 85
      }
    },
  ];

  const filteredPersonnel = securityPersonnel.filter(person => {
    const matchesSearch = 
      person.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      person.badgeNumber.toLowerCase().includes(searchQuery.toLowerCase());
    const matchesStatus = statusFilter === 'all' || person.status === statusFilter;
    return matchesSearch && matchesStatus;
  });

  const selectedPerson = selectedPersonnel 
    ? securityPersonnel.find(p => p.id === selectedPersonnel)
    : null;

  const getStatusColor = (status: SecurityPersonnel['status']) => {
    switch (status) {
      case 'active': return 'bg-green-500';
      case 'responding': return 'bg-amber-500';
      case 'available': return 'bg-blue-500';
      case 'offline': return 'bg-gray-500';
      case 'on-break': return 'bg-purple-500';
      default: return 'bg-gray-500';
    }
  };

  const getStatusText = (status: SecurityPersonnel['status']) => {
    switch (status) {
      case 'active': return 'On Patrol';
      case 'responding': return 'Responding';
      case 'available': return 'Available';
      case 'offline': return 'Off Duty';
      case 'on-break': return 'On Break';
      default: return status;
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h1 className={`text-2xl sm:text-3xl ${vigilClasses.textPrimary}`}>Security Oversight</h1>
          <p className={`text-sm sm:text-base ${vigilClasses.textSecondary}`}>
            Monitor and oversee security personnel activities
          </p>
        </div>
        <div className="flex items-center gap-2 flex-wrap">
          <Button variant="outline" className="gap-2 text-sm sm:text-base">
            <Download className="h-4 w-4" />
            <span className="hidden sm:inline">Export Report</span>
            <span className="sm:hidden">Export</span>
          </Button>
          <Button className="gap-2 text-sm sm:text-base">
            <Eye className="h-4 w-4" />
            <span className="hidden sm:inline">Live Monitoring</span>
            <span className="sm:hidden">Live</span>
          </Button>
        </div>
      </div>

      {/* Stats Overview */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <Card className={vigilClasses.card}>
          <CardContent className="pt-6">
            <div className="flex items-center justify-between">
              <div>
                <p className={`text-sm ${vigilClasses.textSecondary}`}>Assigned Personnel</p>
                <p className={`text-2xl ${vigilClasses.textPrimary}`}>8</p>
              </div>
              <Users className="h-8 w-8 text-blue-500" />
            </div>
          </CardContent>
        </Card>

        <Card className={vigilClasses.card}>
          <CardContent className="pt-6">
            <div className="flex items-center justify-between">
              <div>
                <p className={`text-sm ${vigilClasses.textSecondary}`}>Avg Response Time</p>
                <p className={`text-2xl ${vigilClasses.textPrimary}`}>5.8m</p>
              </div>
              <Clock className="h-8 w-8 text-green-500" />
            </div>
          </CardContent>
        </Card>

        <Card className={vigilClasses.card}>
          <CardContent className="pt-6">
            <div className="flex items-center justify-between">
              <div>
                <p className={`text-sm ${vigilClasses.textSecondary}`}>Active Responses</p>
                <p className={`text-2xl ${vigilClasses.textPrimary}`}>3</p>
              </div>
              <Activity className="h-8 w-8 text-amber-500" />
            </div>
          </CardContent>
        </Card>

        <Card className={vigilClasses.card}>
          <CardContent className="pt-6">
            <div className="flex items-center justify-between">
              <div>
                <p className={`text-sm ${vigilClasses.textSecondary}`}>Verification Rate</p>
                <p className={`text-2xl ${vigilClasses.textPrimary}`}>92%</p>
              </div>
              <UserCheck className="h-8 w-8 text-emerald-500" />
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Main Content Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left Column - Personnel List */}
        <div className="lg:col-span-2">
          <Card className={vigilClasses.card}>
            <CardHeader>
              <div className="flex flex-col gap-4">
                <CardTitle className={vigilClasses.textPrimary}>Security Personnel</CardTitle>
                <div className="flex flex-col sm:flex-row items-stretch sm:items-center gap-3">
                  <div className="relative flex-1">
                    <Search className={`absolute left-2.5 top-2.5 h-4 w-4 ${vigilClasses.textMuted}`} />
                    <Input
                      type="search"
                      placeholder="Search by name or badge..."
                      className="pl-8 w-full"
                      value={searchQuery}
                      onChange={(e) => setSearchQuery(e.target.value)}
                    />
                  </div>
                  <Select value={statusFilter} onValueChange={setStatusFilter}>
                    <SelectTrigger className="w-full sm:w-[180px]">
                      <Filter className="h-4 w-4 mr-2" />
                      <SelectValue placeholder="Filter status" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="all">All Status</SelectItem>
                      <SelectItem value="active">Active</SelectItem>
                      <SelectItem value="responding">Responding</SelectItem>
                      <SelectItem value="available">Available</SelectItem>
                      <SelectItem value="on-break">On Break</SelectItem>
                      <SelectItem value="offline">Offline</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              </div>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {filteredPersonnel.map((person) => (
                  <Card 
                    key={person.id}
                    className={`cursor-pointer transition-all hover:shadow-md ${
                      selectedPersonnel === person.id ? 'ring-2 ring-primary' : ''
                    } ${vigilClasses.card}`}
                    onClick={() => setSelectedPersonnel(person.id)}
                  >
                    <CardContent className="pt-6">
                      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
                        <div className="flex items-center gap-3 sm:gap-4 flex-1 min-w-0">
                          <Avatar className="h-10 w-10 sm:h-12 sm:w-12 flex-shrink-0">
                            <AvatarImage src={person.avatar} />
                            <AvatarFallback className="bg-cyan-500/20 text-cyan-600 dark:text-cyan-400 text-xs sm:text-sm">
                              {person.name.split(' ').map(n => n[0]).join('')}
                            </AvatarFallback>
                          </Avatar>
                          <div className="flex-1 min-w-0">
                            <div className="flex items-center gap-2 flex-wrap">
                              <p className={`${vigilClasses.textPrimary} text-sm sm:text-base truncate`}>{person.name}</p>
                              <Badge variant="outline" className="text-xs">{person.badgeNumber}</Badge>
                            </div>
                            <div className="flex flex-col sm:flex-row sm:items-center gap-1 sm:gap-3 mt-1">
                              <Badge 
                                variant="secondary" 
                                className="flex items-center gap-1 w-fit text-xs"
                              >
                                <div className={`h-2 w-2 rounded-full ${getStatusColor(person.status)}`} />
                                {getStatusText(person.status)}
                              </Badge>
                              <div className={`flex items-center gap-1 text-xs sm:text-sm ${vigilClasses.textSecondary}`}>
                                <MapPin className="h-3 w-3 flex-shrink-0" />
                                <span className="truncate">{person.location}</span>
                              </div>
                            </div>
                          </div>
                        </div>

                        <div className="flex flex-row sm:flex-col items-center sm:items-end justify-between sm:justify-start gap-2 sm:gap-2 flex-shrink-0">
                          {person.currentIncident ? (
                            <div className="text-left sm:text-right flex-1 sm:flex-initial min-w-0">
                              <p className={`text-xs sm:text-sm ${vigilClasses.textPrimary} truncate`}>{person.currentIncident.title}</p>
                              <p className={`text-xs ${vigilClasses.textSecondary}`}>
                                Response: {person.currentIncident.responseTime}min
                              </p>
                            </div>
                          ) : (
                            <p className={`text-xs sm:text-sm ${vigilClasses.textSecondary} flex-1 sm:flex-initial`}>No active incident</p>
                          )}
                          <div className="flex gap-1 sm:gap-2 flex-shrink-0">
                            <Button size="sm" variant="ghost" className="h-8 w-8 p-0">
                              <MessageSquare className="h-4 w-4" />
                            </Button>
                            <Button size="sm" variant="ghost" className="h-8 w-8 p-0">
                              <Phone className="h-4 w-4" />
                            </Button>
                            <Button size="sm" className="h-8 w-8 p-0">
                              <Eye className="h-4 w-4" />
                            </Button>
                          </div>
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                ))}
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Right Column - Selected Personnel Details */}
        <div>
          <Card className={vigilClasses.card}>
            <CardHeader>
              <CardTitle className={vigilClasses.textPrimary}>Personnel Details</CardTitle>
              <CardDescription className={vigilClasses.textSecondary}>
                {selectedPerson ? `Viewing ${selectedPerson.name}` : 'Select a personnel to view details'}
              </CardDescription>
            </CardHeader>
            <CardContent>
              {selectedPerson ? (
                <div className="space-y-6">
                  {/* Profile Header */}
                  <div className="flex items-center gap-3 sm:gap-4">
                    <Avatar className="h-12 w-12 sm:h-16 sm:w-16 flex-shrink-0">
                      <AvatarImage src={selectedPerson.avatar} />
                      <AvatarFallback className="bg-cyan-500/20 text-cyan-600 dark:text-cyan-400">
                        {selectedPerson.name.split(' ').map(n => n[0]).join('')}
                      </AvatarFallback>
                    </Avatar>
                    <div className="flex-1 min-w-0">
                      <h3 className={`text-lg sm:text-xl ${vigilClasses.textPrimary} truncate`}>{selectedPerson.name}</h3>
                      <div className="flex items-center gap-2 mt-1 flex-wrap">
                        <Badge className="text-xs">{selectedPerson.badgeNumber}</Badge>
                        <Badge variant={selectedPerson.assignedOfficer === 'You' ? 'default' : 'secondary'} className="text-xs">
                          <Shield className="h-3 w-3 mr-1" />
                          {selectedPerson.assignedOfficer}
                        </Badge>
                      </div>
                    </div>
                  </div>

                  <Separator />

                  {/* Status & Location */}
                  <div>
                    <h4 className={`mb-2 ${vigilClasses.textPrimary}`}>Current Status</h4>
                    <div className="grid grid-cols-2 gap-4">
                      <div className="space-y-1">
                        <Label className={`text-sm ${vigilClasses.textSecondary}`}>Status</Label>
                        <div className={`flex items-center gap-2 ${vigilClasses.textPrimary}`}>
                          <div className={`h-3 w-3 rounded-full ${getStatusColor(selectedPerson.status)}`} />
                          <span>{getStatusText(selectedPerson.status)}</span>
                        </div>
                      </div>
                      <div className="space-y-1">
                        <Label className={`text-sm ${vigilClasses.textSecondary}`}>Location</Label>
                        <div className={`flex items-center gap-2 ${vigilClasses.textPrimary}`}>
                          <MapPin className="h-4 w-4" />
                          <span>{selectedPerson.location}</span>
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Current Incident */}
                  {selectedPerson.currentIncident && (
                    <>
                      <Separator />
                      <div>
                        <h4 className={`mb-2 ${vigilClasses.textPrimary}`}>Active Response</h4>
                        <Card className={vigilClasses.card}>
                          <CardContent className="pt-4">
                            <div className="space-y-2">
                              <div className="flex justify-between">
                                <span className={`text-sm ${vigilClasses.textSecondary}`}>Incident:</span>
                                <span className={vigilClasses.textPrimary}>{selectedPerson.currentIncident.title}</span>
                              </div>
                              <div className="flex justify-between">
                                <span className={`text-sm ${vigilClasses.textSecondary}`}>Type:</span>
                                <Badge variant="outline">{selectedPerson.currentIncident.type}</Badge>
                              </div>
                              <div className="flex justify-between">
                                <span className={`text-sm ${vigilClasses.textSecondary}`}>Response Time:</span>
                                <span className={vigilClasses.textPrimary}>{selectedPerson.currentIncident.responseTime} minutes</span>
                              </div>
                            </div>
                          </CardContent>
                        </Card>
                      </div>
                    </>
                  )}

                  {/* Performance Stats */}
                  <Separator />
                  <div>
                    <h4 className={`mb-3 text-sm sm:text-base ${vigilClasses.textPrimary}`}>Performance Metrics</h4>
                    <div className="grid grid-cols-2 gap-3 sm:gap-4">
                      <Card className={vigilClasses.card}>
                        <CardContent className="pt-3 sm:pt-4">
                          <p className={`text-xs sm:text-sm ${vigilClasses.textSecondary}`}>Avg Response</p>
                          <p className={`text-xl sm:text-2xl ${vigilClasses.textPrimary}`}>{selectedPerson.performance.avgResponse}m</p>
                        </CardContent>
                      </Card>
                      <Card className={vigilClasses.card}>
                        <CardContent className="pt-3 sm:pt-4">
                          <p className={`text-xs sm:text-sm ${vigilClasses.textSecondary}`}>Incidents</p>
                          <p className={`text-xl sm:text-2xl ${vigilClasses.textPrimary}`}>{selectedPerson.performance.incidentsHandled}</p>
                        </CardContent>
                      </Card>
                    </div>
                    <Card className={`mt-3 sm:mt-4 ${vigilClasses.card}`}>
                      <CardContent className="pt-3 sm:pt-4">
                        <p className={`text-xs sm:text-sm ${vigilClasses.textSecondary}`}>Success Rate</p>
                        <div className="flex items-center justify-between gap-4">
                          <p className={`text-xl sm:text-2xl ${vigilClasses.textPrimary}`}>{selectedPerson.performance.successRate}%</p>
                          <div className="flex-1 max-w-[100px] h-2 bg-gray-200 dark:bg-gray-800 rounded-full overflow-hidden">
                            <div 
                              className="h-full bg-green-500"
                              style={{ width: `${selectedPerson.performance.successRate}%` }}
                            />
                          </div>
                        </div>
                      </CardContent>
                    </Card>
                  </div>

                  {/* Action Buttons */}
                  <div className="grid grid-cols-2 gap-2 sm:gap-3">
                    <Button variant="outline" className="gap-1 sm:gap-2 text-xs sm:text-sm">
                      <MessageSquare className="h-3 w-3 sm:h-4 sm:w-4" />
                      <span className="hidden sm:inline">Message</span>
                      <span className="sm:hidden">Chat</span>
                    </Button>
                    <Button className="gap-1 sm:gap-2 text-xs sm:text-sm">
                      <CheckCircle className="h-3 w-3 sm:h-4 sm:w-4" />
                      <span className="hidden sm:inline">Verify Response</span>
                      <span className="sm:hidden">Verify</span>
                    </Button>
                    <Button variant="outline" className="gap-1 sm:gap-2 text-xs sm:text-sm">
                      <Phone className="h-3 w-3 sm:h-4 sm:w-4" />
                      Call
                    </Button>
                    <Button variant="outline" className="gap-1 sm:gap-2 text-xs sm:text-sm">
                      <AlertCircle className="h-3 w-3 sm:h-4 sm:w-4" />
                      <span className="hidden sm:inline">Report Issue</span>
                      <span className="sm:hidden">Report</span>
                    </Button>
                  </div>
                </div>
              ) : (
                <div className={`text-center py-8 ${vigilClasses.textSecondary}`}>
                  <Users className="h-12 w-12 mx-auto mb-4 opacity-20" />
                  <p>Select a security personnel from the list to view details</p>
                </div>
              )}
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}
