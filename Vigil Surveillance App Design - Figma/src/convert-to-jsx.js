#!/usr/bin/env node

/**
 * TypeScript to JavaScript Converter
 * 
 * This script converts .tsx/.ts files to .jsx/.js files
 * Run: node convert-to-jsx.js
 */

const fs = require('fs');
const path = require('path');

// Directories to convert
const DIRS_TO_CONVERT = ['./components', './hooks', './utils'];
const OUTPUT_DIR = './src';

// Conversion statistics
let stats = {
  converted: 0,
  skipped: 0,
  errors: 0,
};

/**
 * Clean TypeScript syntax from code
 */
function convertTypeScriptToJavaScript(content, filename) {
  let result = content;

  // Remove type imports (import type { ... })
  result = result.replace(/import\s+type\s*\{[^}]+\}\s*from\s+['"][^'"]+['"];?\n?/g, '');
  
  // Remove specific types from imports
  result = result.replace(/import\s*\{([^}]+)\}\s*from/g, (match, imports) => {
    const cleanedImports = imports
      .split(',')
      .filter(imp => !imp.includes('type '))
      .join(',');
    return `import {${cleanedImports}} from`;
  });

  // Remove React.FC, React.ReactElement, etc.
  result = result.replace(/:\s*React\.(FC|ReactElement|ReactNode|JSX\.Element)(<[^>]+>)?/g, '');
  
  // Remove interface declarations
  result = result.replace(/^interface\s+\w+\s*\{[^}]*\}\s*$/gm, '');
  result = result.replace(/interface\s+\w+\s*\{[^}]+\}/gs, '');
  
  // Remove type declarations
  result = result.replace(/^type\s+\w+\s*=\s*[^;]+;\s*$/gm, '');
  result = result.replace(/^export\s+type\s+\w+\s*=\s*[^;]+;\s*$/gm, '');
  
  // Remove function parameter types
  result = result.replace(/(\w+)\s*:\s*[\w<>[\]|&\s]+(?=[,)])/g, '$1');
  
  // Remove function return types
  result = result.replace(/\)\s*:\s*[\w<>[\]|&\s.]+\s*(\{|=>)/g, ') $1');
  
  // Remove variable type annotations
  result = result.replace(/:\s*[\w<>[\]|&\s]+(\s*=)/g, '$1');
  
  // Remove useState generic types
  result = result.replace(/useState<[^>]+>/g, 'useState');
  result = result.replace(/useRef<[^>]+>/g, 'useRef');
  result = result.replace(/useCallback<[^>]+>/g, 'useCallback');
  result = result.replace(/useMemo<[^>]+>/g, 'useMemo');
  
  // Remove 'as' type assertions
  result = result.replace(/\s+as\s+[\w<>[\]|&\s]+/g, '');
  
  // Remove generic type parameters from components
  result = result.replace(/<[A-Z]\w*>/g, '');
  
  // Remove enum declarations
  result = result.replace(/enum\s+\w+\s*\{[^}]+\}/gs, '');
  
  // Clean up multiple empty lines
  result = result.replace(/\n{3,}/g, '\n\n');
  
  return result;
}

/**
 * Update import paths from .tsx to .jsx
 */
function updateImportPaths(content) {
  // Update relative imports
  content = content.replace(/from\s+['"](\.[^'"]+)\.tsx['"]/g, 'from \'$1.jsx\'');
  content = content.replace(/from\s+['"](\.[^'"]+)\.ts['"]/g, 'from \'$1.js\'');
  
  // Handle imports without extension (add .jsx/.js)
  content = content.replace(/from\s+['"](\.\/components\/[^'"]+)(?<!\.jsx|\.js)['"]/g, 'from \'$1.jsx\'');
  content = content.replace(/from\s+['"](\.\/hooks\/[^'"]+)(?<!\.jsx|\.js)['"]/g, 'from \'$1.js\'');
  content = content.replace(/from\s+['"](\.\/utils\/[^'"]+)(?<!\.jsx|\.js)['"]/g, 'from \'$1.js\'');
  
  return content;
}

/**
 * Process a single file
 */
function processFile(filePath, relativePath) {
  try {
    const content = fs.readFileSync(filePath, 'utf8');
    
    // Convert TypeScript to JavaScript
    let converted = convertTypeScriptToJavaScript(content, path.basename(filePath));
    converted = updateImportPaths(converted);
    
    // Determine output path
    const ext = path.extname(filePath);
    const newExt = ext === '.tsx' ? '.jsx' : '.js';
    const outputPath = path.join(OUTPUT_DIR, relativePath.replace(ext, newExt));
    
    // Create directory if it doesn't exist
    const outputDir = path.dirname(outputPath);
    if (!fs.existsSync(outputDir)) {
      fs.mkdirSync(outputDir, { recursive: true });
    }
    
    // Write converted file
    fs.writeFileSync(outputPath, converted, 'utf8');
    
    console.log(`✅ Converted: ${relativePath} -> ${outputPath.replace('./', '')}`);
    stats.converted++;
    
  } catch (error) {
    console.error(`❌ Error converting ${filePath}:`, error.message);
    stats.errors++;
  }
}

/**
 * Process directory recursively
 */
function processDirectory(dirPath, baseDir) {
  if (!fs.existsSync(dirPath)) {
    console.log(`⚠️  Directory not found: ${dirPath}`);
    return;
  }

  const items = fs.readdirSync(dirPath);
  
  items.forEach(item => {
    const fullPath = path.join(dirPath, item);
    const stat = fs.statSync(fullPath);
    
    if (stat.isDirectory()) {
      processDirectory(fullPath, baseDir);
    } else if (item.endsWith('.tsx') || item.endsWith('.ts')) {
      // Skip declaration files
      if (item.endsWith('.d.ts')) {
        console.log(`⏭️  Skipped: ${item} (declaration file)`);
        stats.skipped++;
        return;
      }
      
      const relativePath = path.relative(baseDir, fullPath);
      processFile(fullPath, relativePath);
    }
  });
}

/**
 * Main conversion process
 */
function main() {
  console.log('🚀 Starting TypeScript to JavaScript conversion...\n');
  
  // Create output directory
  if (!fs.existsSync(OUTPUT_DIR)) {
    fs.mkdirSync(OUTPUT_DIR, { recursive: true });
  }
  
  // Process each directory
  DIRS_TO_CONVERT.forEach(dir => {
    console.log(`\n📁 Processing directory: ${dir}`);
    processDirectory(dir, path.dirname(dir));
  });
  
  // Copy styles
  console.log('\n📁 Copying styles directory...');
  if (fs.existsSync('./styles')) {
    const stylesOutput = path.join(OUTPUT_DIR, 'styles');
    if (!fs.existsSync(stylesOutput)) {
      fs.mkdirSync(stylesOutput, { recursive: true });
    }
    const styles = fs.readdirSync('./styles');
    styles.forEach(file => {
      if (file.endsWith('.css')) {
        fs.copyFileSync(
          path.join('./styles', file),
          path.join(stylesOutput, file)
        );
        console.log(`✅ Copied: styles/${file}`);
      }
    });
  }
  
  // Print statistics
  console.log('\n' + '='.repeat(50));
  console.log('📊 Conversion Summary:');
  console.log('='.repeat(50));
  console.log(`✅ Files converted: ${stats.converted}`);
  console.log(`⏭️  Files skipped: ${stats.skipped}`);
  console.log(`❌ Errors: ${stats.errors}`);
  console.log('='.repeat(50));
  
  if (stats.errors === 0) {
    console.log('\n✨ Conversion completed successfully!');
    console.log('\n📝 Next steps:');
    console.log('   1. Review the converted files in /src directory');
    console.log('   2. Run: npm install');
    console.log('   3. Run: npm start');
    console.log('   4. Test the application thoroughly');
  } else {
    console.log('\n⚠️  Conversion completed with errors. Please review the errors above.');
  }
}

// Run the conversion
main();
