import { readdir, stat } from 'fs/promises';
import { existsSync } from 'fs';
import { join } from 'path';
import { homedir } from 'os';

/**
 * Metadata for a discovered transcript file
 */
export interface TranscriptFile {
  projectPath: string;        // Decoded original path (e.g., /home/bellman/Workspace/foo)
  projectDir: string;         // Encoded directory name (e.g., -home-bellman-Workspace-foo)
  sessionId: string;          // UUID from filename
  filePath: string;           // Full path to .jsonl
  fileSize: number;           // Bytes
  modifiedTime: Date;
}

/**
 * Result of scanning for transcripts
 */
export interface ScanResult {
  transcripts: TranscriptFile[];
  totalSize: number;
  projectCount: number;
}

/**
 * Options for scanning transcripts
 */
export interface ScanOptions {
  projectFilter?: string;     // Glob pattern for project path
  minDate?: Date;             // Only files modified after this date
}

/**
 * UUID regex pattern for session IDs
 */
const UUID_REGEX = /^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$/;

/**
 * Decode project directory name back to original path.
 *
 * The encoding scheme used by Claude Code is lossy - it converts all path
 * separators (/) to dashes (-), but legitimate dashes in directory names
 * also become dashes, making them indistinguishable.
 *
 * Strategy:
 * 1. Try simple decode (all dashes -> slashes) and check if path exists
 * 2. If not, try to reconstruct by checking filesystem for partial matches
 * 3. Fall back to simple decode if nothing else works
 *
 * Example: "-home-bellman-my-project"
 *   - Simple decode: "/home/bellman/my/project" (WRONG if "my-project" is one dir)
 *   - Smart decode: "/home/bellman/my-project" (checks filesystem)
 *
 * @internal Exported for testing
 */
export function decodeProjectPath(dirName: string): string {
  if (!dirName.startsWith('-')) {
    return dirName;
  }

  // Simple decode: replace all dashes with slashes
  const simplePath = '/' + dirName.slice(1).replace(/-/g, '/');

  // If simple decode exists, we're done
  if (existsSync(simplePath)) {
    return simplePath;
  }

  // Try to reconstruct by checking filesystem for partial matches
  const segments = dirName.slice(1).split('-');
  const possiblePaths: string[] = [];

  // Generate all possible interpretations by trying different hyphen positions
  function generatePaths(parts: string[], index: number, currentPath: string): void {
    if (index >= parts.length) {
      possiblePaths.push(currentPath);
      return;
    }

    // Try adding next segment as a new directory
    generatePaths(parts, index + 1, currentPath + '/' + parts[index]);

    // Try combining with previous segment using hyphen (if not first segment)
    if (index > 0 && currentPath) {
      const pathParts = currentPath.split('/');
      const lastPart = pathParts.pop() || '';
      const newPath = pathParts.join('/') + '/' + lastPart + '-' + parts[index];
      generatePaths(parts, index + 1, newPath);
    }
  }

  generatePaths(segments, 0, '');

  // Find the first path that exists on filesystem
  for (const path of possiblePaths) {
    if (existsSync(path)) {
      return path;
    }
  }

  // Fall back to simple decode
  return simplePath;
}

/**
 * Check if a path matches a glob pattern (simple implementation)
 */
function matchesPattern(path: string, pattern?: string): boolean {
  if (!pattern) return true;

  // Convert glob pattern to regex
  const regexPattern = pattern
    .replace(/\./g, '\\.')
    .replace(/\*/g, '.*')
    .replace(/\?/g, '.');

  const regex = new RegExp(`^${regexPattern}$`);
  return regex.test(path);
}

/**
 * Scan for all transcript files in ~/.claude/projects/
 */
export async function scanTranscripts(options: ScanOptions = {}): Promise<ScanResult> {
  const projectsDir = join(homedir(), '.claude', 'projects');
  const transcripts: TranscriptFile[] = [];
  const projectDirs = new Set<string>();

  try {
    // Read all project directories
    const entries = await readdir(projectsDir, { withFileTypes: true });

    for (const entry of entries) {
      if (!entry.isDirectory()) continue;

      const projectDir = entry.name;
      const projectPath = decodeProjectPath(projectDir);

      // Apply project filter if specified
      if (!matchesPattern(projectPath, options.projectFilter)) {
        continue;
      }

      const fullProjectPath = join(projectsDir, projectDir);

      // Read all files in this project directory
      const projectFiles = await readdir(fullProjectPath);

      for (const fileName of projectFiles) {
        // Skip sessions-index.json and any non-.jsonl files
        if (fileName === 'sessions-index.json' || !fileName.endsWith('.jsonl')) {
          continue;
        }

        // Extract session ID from filename
        const sessionId = fileName.replace('.jsonl', '');

        // Validate session ID format (must be UUID)
        if (!UUID_REGEX.test(sessionId)) {
          continue;
        }

        const filePath = join(fullProjectPath, fileName);
        const fileStats = await stat(filePath);

        // Apply date filter if specified
        if (options.minDate && fileStats.mtime < options.minDate) {
          continue;
        }

        transcripts.push({
          projectPath,
          projectDir,
          sessionId,
          filePath,
          fileSize: fileStats.size,
          modifiedTime: fileStats.mtime
        });

        projectDirs.add(projectDir);
      }
    }
  } catch (error) {
    // If projects directory doesn't exist, return empty result
    if ((error as NodeJS.ErrnoException).code === 'ENOENT') {
      return {
        transcripts: [],
        totalSize: 0,
        projectCount: 0
      };
    }
    throw error;
  }

  const totalSize = transcripts.reduce((sum, t) => sum + t.fileSize, 0);

  return {
    transcripts,
    totalSize,
    projectCount: projectDirs.size
  };
}
