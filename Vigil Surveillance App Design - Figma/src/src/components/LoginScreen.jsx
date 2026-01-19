import { useState } from 'react';
import { Button } from './ui/button';
import { Input } from './ui/input';
import { Label } from './ui/label';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './ui/tabs';
import { Shield, Eye, Lock } from 'lucide-react';
import { toast } from 'sonner@2.0.3';
import { VigilLogo } from './VigilLogo';
import { ThemeToggle } from './ThemeToggle';
import { useTheme } from './ThemeProvider';
import { AnimatedBackgroundSimple } from './AnimatedBackgroundSimple';

export function LoginScreen({ onLogin }) {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const { theme } = useTheme();

  const handleLogin = (role) => {
    if (!email || !password) {
      toast.error('Please enter email and password');
      return;
    }
    toast.success(`Logged in as ${role}`);
    onLogin(role);
  };

  return (
    <AnimatedBackgroundSimple mode={theme === 'dark' ? 'dark' : 'light'}>
      <div className="min-h-screen flex items-center justify-center p-4 relative">
        {/* Content Layer */}
        <div className="relative z-10 w-full max-w-md">
          {/* Theme Toggle - Top Right */}
          <div className="fixed top-6 right-6 z-20">
            <ThemeToggle />
          </div>

          {/* Logo and Header */}
          <div className="text-center mb-8">
            <div className="inline-flex items-center justify-center mb-4">
              <VigilLogo width={200} height={67} theme={theme} />
            </div>
            <p className="text-gray-600 dark:text-gray-400">AI-Powered Surveillance System</p>
          </div>

          <Card className="border-gray-200 dark:border-gray-800 bg-white/80 dark:bg-[#13182b]/80 backdrop-blur-xl shadow-2xl">
            <CardHeader>
              <CardTitle className="text-gray-900 dark:text-white">Sign In</CardTitle>
              <CardDescription>Choose your role and sign in to continue</CardDescription>
            </CardHeader>
            <CardContent>
              <Tabs defaultValue="admin" className="w-full">
                <TabsList className="grid w-full grid-cols-3 bg-gray-100 dark:bg-[#0a0e1a]">
                  <TabsTrigger value="admin" className="data-[state=active]:bg-blue-500 dark:data-[state=active]:bg-cyan-500 data-[state=active]:text-gray-900 dark:data-[state=active]:text-white">
                    Admin
                  </TabsTrigger>
                  <TabsTrigger value="officer" className="data-[state=active]:bg-blue-500 dark:data-[state=active]:bg-cyan-500 data-[state=active]:text-gray-900 dark:data-[state=active]:text-white">
                    Officer
                  </TabsTrigger>
                  <TabsTrigger value="security" className="data-[state=active]:bg-blue-500 dark:data-[state=active]:bg-cyan-500 data-[state=active]:text-gray-900 dark:data-[state=active]:text-white">
                    Security
                  </TabsTrigger>
                </TabsList>

                <TabsContent value="admin" className="space-y-4">
                  <LoginForm
                    role="admin"
                    icon={<Shield className="w-5 h-5" />}
                    email={email}
                    password={password}
                    onEmailChange={setEmail}
                    onPasswordChange={setPassword}
                    onSubmit={() => handleLogin('admin')}
                  />
                </TabsContent>

                <TabsContent value="officer" className="space-y-4">
                  <LoginForm
                    role="officer"
                    icon={<Eye className="w-5 h-5" />}
                    email={email}
                    password={password}
                    onEmailChange={setEmail}
                    onPasswordChange={setPassword}
                    onSubmit={() => handleLogin('officer')}
                  />
                </TabsContent>

                <TabsContent value="security" className="space-y-4">
                  <LoginForm
                    role="security"
                    icon={<Lock className="w-5 h-5" />}
                    email={email}
                    password={password}
                    onEmailChange={setEmail}
                    onPasswordChange={setPassword}
                    onSubmit={() => handleLogin('security')}
                  />
                </TabsContent>
              </Tabs>
            </CardContent>
          </Card>

          <p className="text-center text-gray-500 text-sm mt-6">
            Demo: Use any email and password to login
          </p>
        </div>
      </div>
    </AnimatedBackgroundSimple>
  );
}

function LoginForm({ role, icon, email, password, onEmailChange, onPasswordChange, onSubmit }) {
  return (
    <form
      onSubmit={(e) => {
        e.preventDefault();
        onSubmit();
      }}
      className="space-y-4"
    >
      <div className="flex items-center gap-2 text-blue-500 dark:text-cyan-400 mb-4">
        {icon}
        <span className="capitalize">Sign in as {role}</span>
      </div>

      <div className="space-y-2">
        <Label htmlFor={`email-${role}`} className="text-gray-700 dark:text-gray-300">
          Email
        </Label>
        <Input
          id={`email-${role}`}
          type="email"
          placeholder={`${role}@vigil.com`}
          value={email}
          onChange={(e) => onEmailChange(e.target.value)}
          className="bg-gray-50 dark:bg-[#0a0e1a] border-gray-300 dark:border-gray-700 text-gray-900 dark:text-white placeholder:text-gray-400 dark:placeholder:text-gray-500"
        />
      </div>

      <div className="space-y-2">
        <Label htmlFor={`password-${role}`} className="text-gray-700 dark:text-gray-300">
          Password
        </Label>
        <Input
          id={`password-${role}`}
          type="password"
          placeholder="Enter your password"
          value={password}
          onChange={(e) => onPasswordChange(e.target.value)}
          className="bg-gray-50 dark:bg-[#0a0e1a] border-gray-300 dark:border-gray-700 text-gray-900 dark:text-white placeholder:text-gray-400 dark:placeholder:text-gray-500"
        />
      </div>

      <Button type="submit" className="w-full bg-blue-500 hover:bg-blue-600 dark:bg-cyan-500 dark:hover:bg-cyan-600 text-gray-900 dark:text-white">
        Sign In
      </Button>
    </form>
  );
}
