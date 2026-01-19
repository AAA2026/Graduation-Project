# TypeScript to JavaScript Conversion Guide

This guide will help you complete the conversion from TypeScript (.tsx) to JavaScript (.jsx) files.

## Automated Conversion Script

Create and run this Node.js script to convert all remaining files:

```javascript
// convert-to-jsx.js
const fs = require('fs');
const path = require('path');

function convertFile(filePath) {
  let content = fs.readFileSync(filePath, 'utf8');
  
  // Remove type imports
  content = content.replace(/import\s+type\s+\{[^}]+\}\s+from\s+['"][^'"]+['"];?\n?/g, '');
  
  // Remove type annotations from function parameters
  content = content.replace(/(\w+):\s*\w+(\[\])?(\s*\|[^,)]+)?/g, '$1');
  
  // Remove return type annotations
  content = content.replace(/\):\s*\w+(\[\])?(\s*\|[^{]+)?\s*\{/g, ') {');
  content = content.replace(/\):\s*JSX\.Element\s*\{/g, ') {');
  content = content.replace(/\):\s*React\.ReactElement\s*\{/g, ') {');
  
  // Remove interface declarations
  content = content.replace(/interface\s+\w+\s*\{[^}]+\}\n*/g, '');
  
  // Remove type declarations
  content = content.replace(/type\s+\w+\s*=\s*[^;]+;\n*/g, '');
  
  // Remove generic type parameters
  content = content.replace(/<\w+(\[\])?>/g, '');
  
  // Remove 'as' type assertions
  content = content.replace(/\s+as\s+\w+/g, '');
  
  return content;
}

function processDirectory(dirPath) {
  const items = fs.readdirSync(dirPath);
  
  items.forEach(item => {
    const fullPath = path.join(dirPath, item);
    const stat = fs.statSync(fullPath);
    
    if (stat.isDirectory()) {
      processDirectory(fullPath);
    } else if (item.endsWith('.tsx') || item.endsWith('.ts')) {
      // Skip .d.ts files
      if (item.endsWith('.d.ts')) return;
      
      const content = convertFile(fullPath);
      const newPath = fullPath.replace(/\.tsx?$/, item.endsWith('.tsx') ? '.jsx' : '.js');
      
      fs.writeFileSync(newPath, content);
      console.log(`Converted: ${fullPath} -> ${newPath}`);
      
      // Optionally delete the original file
      // fs.unlinkSync(fullPath);
    }
  });
}

// Run conversion
processDirectory('./src');
console.log('Conversion complete!');
```

## Manual Conversion Steps

For each .tsx file, follow these steps:

### 1. Remove Type Annotations

**Before:**
```typescript
const [count, setCount] = useState<number>(0);
function handleClick(event: React.MouseEvent): void {
```

**After:**
```javascript
const [count, setCount] = useState(0);
function handleClick(event) {
```

### 2. Remove Interfaces

**Before:**
```typescript
interface Props {
  name: string;
  age: number;
}

function Component({ name, age }: Props) {
```

**After:**
```javascript
function Component({ name, age }) {
```

### 3. Remove Type Imports

**Before:**
```typescript
import type { FC, ReactNode } from 'react';
import { useState, type Dispatch } from 'react';
```

**After:**
```javascript
import { useState } from 'react';
```

### 4. Remove Enums

**Before:**
```typescript
enum Status {
  Active = 'active',
  Inactive = 'inactive'
}
```

**After:**
```javascript
const Status = {
  Active: 'active',
  Inactive: 'inactive'
};
```

### 5. Update File Extensions

- Rename `.tsx` → `.jsx`
- Rename `.ts` → `.js`
- Update all imports to use new extensions

## Files Already Converted

✅ `/src/index.js` - Entry point
✅ `/src/App.jsx` - Main app component  
✅ `/src/components/ThemeProvider.jsx` - Theme context

## Files Needing Conversion

Priority order:

### High Priority (Core Components)
- [ ] `/src/components/ModernSecurityLayout.jsx`
- [ ] `/src/components/LoginScreen.jsx`
- [ ] `/src/components/AdminDashboard.jsx`
- [ ] `/src/components/OfficerDashboard.jsx`
- [ ] `/src/components/SecurityAuthorityApp.jsx`

### Medium Priority (Feature Components)
- [ ] `/src/components/DVRCameraGrid.jsx`
- [ ] `/src/components/LiveCameraGrid.jsx`
- [ ] `/src/components/MockCameraFeed.jsx`
- [ ] `/src/components/IncidentDetailModal.jsx`
- [ ] `/src/components/VideoModal.jsx`
- [ ] `/src/components/VigilLogo.jsx`
- [ ] `/src/components/AnimatedBackgroundSimple.jsx`

### Low Priority (UI Components)
- [ ] All files in `/src/components/ui/`
- [ ] `/src/components/hooks/` files
- [ ] `/src/hooks/` files
- [ ] `/src/utils/` files

## Quick Find & Replace

Use your editor's find-and-replace (Ctrl/Cmd + Shift + H):

### 1. Remove simple type annotations
Find: `: (string|number|boolean|any)`
Replace: (empty)

### 2. Remove React.FC
Find: `: React\.FC<.*?>`
Replace: (empty)

### 3. Remove useState types
Find: `useState<(.*?)>`
Replace: `useState`

### 4. Fix imports
Find: `from ['"]\.\/(.*)\.tsx?['"]`
Replace: `from './$1.jsx'`

## Common Patterns

### Props Destructuring

**Before:**
```typescript
interface ComponentProps {
  title: string;
  count: number;
  onClose: () => void;
}

export function Component({ title, count, onClose }: ComponentProps) {
```

**After:**
```javascript
export function Component({ title, count, onClose }) {
```

### useState Hook

**Before:**
```typescript
const [data, setData] = useState<DataType | null>(null);
```

**After:**
```javascript
const [data, setData] = useState(null);
```

### Event Handlers

**Before:**
```typescript
const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
```

**After:**
```javascript
const handleChange = (e) => {
```

### Component Export

**Before:**
```typescript
export const Component: React.FC<Props> = ({ prop1, prop2 }) => {
```

**After:**
```javascript
export const Component = ({ prop1, prop2 }) => {
```

## Validation

After conversion, check:

1. **No build errors**: `npm start`
2. **App runs**: Open http://localhost:3000
3. **Login works**: Test all three roles
4. **Features work**: Test cameras, incidents, theme toggle
5. **No console errors**: Check browser console

## Tips

- Convert files in order of dependency (leaf components first)
- Test frequently during conversion
- Keep a backup of original files
- Use version control (git) to track changes
- Run linter after conversion: `npm run lint` (if configured)

## Need Help?

If you encounter errors:

1. **Check import paths**: Ensure all imports use `.jsx` or `.js`
2. **Verify exports**: Make sure default/named exports match
3. **Console logs**: Add `console.log()` to debug
4. **Rollback**: Use git to revert problematic changes

---

**Tip**: Start with leaf components (those that don't import other custom components) and work your way up to App.jsx.
