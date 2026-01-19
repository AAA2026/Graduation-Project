import { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Button } from './ui/button';
import { Badge } from './ui/badge';
import { AnimatedBackground } from './AnimatedBackground';
import { Zap, Eye, Settings } from 'lucide-react';

export function AnimatedBackgroundDemo() {
  const [intensity, setIntensity] = useState<'subtle' | 'medium' | 'strong'>('subtle');

  return (
    <div className="space-y-6 p-6">
      {/* Controls */}
      <Card className="bg-card/80 backdrop-blur border-border">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Settings className="w-5 h-5" />
            Animated Background Controls
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex gap-2">
            <Button
              size="sm"
              variant={intensity === 'subtle' ? 'default' : 'outline'}
              onClick={() => setIntensity('subtle')}
            >
              Subtle
            </Button>
            <Button
              size="sm"
              variant={intensity === 'medium' ? 'default' : 'outline'}
              onClick={() => setIntensity('medium')}
            >
              Medium
            </Button>
            <Button
              size="sm"
              variant={intensity === 'strong' ? 'default' : 'outline'}
              onClick={() => setIntensity('strong')}
            >
              Strong
            </Button>
          </div>
        </CardContent>
      </Card>

      {/* Preview Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Dashboard Variant */}
        <Card className="relative overflow-hidden border-border min-h-[400px]">
          <AnimatedBackground variant="dashboard" intensity={intensity} />
          <div className="relative z-10 p-6">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Eye className="w-5 h-5" />
                Dashboard Background
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <p className="text-muted-foreground">
                This animated background provides a dynamic, professional feel for your dashboard.
              </p>
              <div className="space-y-2">
                <Badge className="bg-cyan-500/20 text-cyan-400 border-cyan-500/30">
                  Animated Gradients
                </Badge>
                <Badge className="bg-blue-500/20 text-blue-400 border-blue-500/30">
                  Floating Shapes
                </Badge>
                <Badge className="bg-purple-500/20 text-purple-400 border-purple-500/30">
                  Theme Aware
                </Badge>
              </div>
            </CardContent>
          </div>
        </Card>

        {/* Modal Variant */}
        <Card className="relative overflow-hidden border-border min-h-[400px]">
          <AnimatedBackground variant="modal" intensity={intensity} />
          <div className="relative z-10 p-6">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Zap className="w-5 h-5" />
                Modal Background
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <p className="text-muted-foreground">
                Perfect for modals and overlays with subtle border glow.
              </p>
              <div className="space-y-2">
                <Badge className="bg-green-500/20 text-green-400 border-green-500/30">
                  Border Glow
                </Badge>
                <Badge className="bg-amber-500/20 text-amber-400 border-amber-500/30">
                  Noise Texture
                </Badge>
                <Badge className="bg-red-500/20 text-red-400 border-red-500/30">
                  Vignette Effect
                </Badge>
              </div>
            </CardContent>
          </div>
        </Card>
      </div>

      {/* Feature List */}
      <Card className="bg-card/80 backdrop-blur border-border">
        <CardHeader>
          <CardTitle>Features</CardTitle>
        </CardHeader>
        <CardContent>
          <ul className="space-y-2 text-sm text-muted-foreground">
            <li className="flex items-start gap-2">
              <span className="text-cyan-400">✓</span>
              <span><strong>Theme Aware:</strong> Automatically adapts colors for dark and light modes</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-cyan-400">✓</span>
              <span><strong>Smooth Animations:</strong> Uses Framer Motion for fluid 20-30s animation loops</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-cyan-400">✓</span>
              <span><strong>Performance Optimized:</strong> GPU-accelerated with CSS transforms</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-cyan-400">✓</span>
              <span><strong>Adjustable Intensity:</strong> Three intensity levels (subtle, medium, strong)</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-cyan-400">✓</span>
              <span><strong>Accessibility:</strong> Maintains high contrast for content readability</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-cyan-400">✓</span>
              <span><strong>Multiple Variants:</strong> Dashboard, modal, and card variants</span>
            </li>
          </ul>
        </CardContent>
      </Card>

      {/* Implementation Guide */}
      <Card className="bg-card/80 backdrop-blur border-border">
        <CardHeader>
          <CardTitle>Usage Examples</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <div>
              <h3 className="font-semibold text-sm mb-2">Dashboard Background:</h3>
              <code className="block bg-muted p-3 rounded text-xs">
                {`<AnimatedBackground variant="dashboard" intensity="subtle" />`}
              </code>
            </div>
            <div>
              <h3 className="font-semibold text-sm mb-2">Modal Background:</h3>
              <code className="block bg-muted p-3 rounded text-xs">
                {`<AnimatedBackground variant="modal" intensity="medium" />`}
              </code>
            </div>
            <div>
              <h3 className="font-semibold text-sm mb-2">Card Background:</h3>
              <code className="block bg-muted p-3 rounded text-xs">
                {`<AnimatedBackground variant="card" intensity="subtle" className="rounded-lg" />`}
              </code>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
