import { spawn } from 'node:child_process'
import { fileURLToPath } from 'node:url'
import path from 'node:path'
import fs from 'node:fs'

export const name = 'dsh-sam2matting'

export const inject = ['tools']

const REPO_ROOT = path.dirname(fileURLToPath(import.meta.url))
const MAX_CAPTURE = 262144

function findPython() {
  const candidates = [
    process.env.SAM2MATTING_PYTHON,
    process.platform === 'win32'
      ? path.join(REPO_ROOT, 'venv', 'Scripts', 'python.exe')
      : path.join(REPO_ROOT, 'venv', 'bin', 'python'),
    'python',
    'python3',
  ].filter(Boolean)
  for (const candidate of candidates) {
    if (!candidate.includes(path.sep) || fs.existsSync(candidate)) {
      return candidate
    }
  }
  return 'python'
}

function runPython(exec, script, args) {
  return new Promise((resolve) => {
    const python = findPython()
    const child = spawn(python, [script, ...args], {
      cwd: REPO_ROOT,
      env: process.env,
      windowsHide: true,
    })
    let out = ''
    let err = ''
    const onAbort = () => { try { child.kill() } catch { /* already dead */ } }
    exec.signal?.addEventListener('abort', onAbort)
    const cap = (chunk, acc) => {
      acc += chunk.toString()
      return acc.length > MAX_CAPTURE ? acc.slice(-MAX_CAPTURE) : acc
    }
    child.stdout.on('data', (d) => { out = cap(d, out) })
    child.stderr.on('data', (d) => { err = cap(d, err) })
    child.on('close', (code) => {
      exec.signal?.removeEventListener('abort', onAbort)
      resolve(code === 0
        ? (out.trim() || '(no output)')
        : `[sam2matting exit ${code}]\n${(out + '\n' + err).trim() || '(no output)'}`)
    })
    child.on('error', (e) => {
      exec.signal?.removeEventListener('abort', onAbort)
      resolve(`Failed to start Python (${python}): ${e.message}\n` +
        'Install Python 3.10+ with CUDA torch (or set SAM2MATTING_PYTHON) and ' +
        'make sure checkpoints are present.')
    })
  })
}

export function apply(ctx) {
  ctx.tools.register({
    name: 'sam2matting_batch',
    description:
      'Run SAM2Matting on a video file, a directory of frames, or a single image. ' +
      'Writes three output folders (alpha/, composite/, transparent/) with per-frame ' +
      'mattes. The first-frame prompt mask is generated automatically with rembg ' +
      '(u2net), or supplied via the mask argument. Requires a CUDA GPU. ' +
      'Long-running: one call per job. Variants: sam2.1tiny (fast), sam2.1base+ ' +
      '(default, higher quality), sam3 (best quality, large checkpoint).',
    parameters: {
      type: 'object',
      properties: {
        input: {
          type: 'string',
          description: 'Absolute path to a video file, frames directory, or single image',
        },
        output: {
          type: 'string',
          description: 'Output root directory. Default: <input>_<timestamp>_matting next to the input',
        },
        variant: {
          type: 'string',
          enum: ['sam2.1tiny', 'sam2.1base+', 'sam3'],
          description: 'Matting model variant (default sam2.1base+)',
        },
        background: {
          type: 'string',
          description: 'Composite background color as "R,G,B" (default "0,0,0")',
        },
        mask: {
          type: 'string',
          description: 'Optional first-frame mask PNG (white = foreground). Skips automatic rembg masking',
        },
      },
      required: ['input'],
    },
    output: {
      schema: { type: 'string' },
      render: (_args, value) => [{ type: 'text', text: value }],
    },
    async execute(args, exec) {
      const argv = ['--input', args.input]
      if (args.output) argv.push('--output', args.output)
      if (args.variant) argv.push('--variant', args.variant)
      if (args.background) argv.push('--bg', args.background)
      if (args.mask) argv.push('--mask', args.mask)
      return runPython(exec, 'batch_matting.py', argv)
    },
  })

  ctx.tools.register({
    name: 'sam2matting_interactive',
    description:
      'Interactive video matting driven by an explicit prompt on the first frame: ' +
      'a point ("x,y" coordinates) or a box ("x1,y1,x2,y2"), or for SAM3 a text ' +
      'prompt (e.g. "girl"). The mask is propagated through the video with the ' +
      'SAM2/SAM3 video predictor. Output goes to the chosen output_dir. ' +
      'Requires a CUDA GPU.',
    parameters: {
      type: 'object',
      properties: {
        video_dir: {
          type: 'string',
          description: 'Directory of input frames (png/jpg)',
        },
        output_dir: {
          type: 'string',
          description: 'Directory for output files (default output_video)',
        },
        model: {
          type: 'string',
          enum: ['sam2', 'sam3'],
          description: 'Which SAM version to use (default sam2)',
        },
        prompt_type: {
          type: 'string',
          enum: ['point', 'box', 'language'],
          description: 'Prompt kind: point, box, or language (language is SAM3 only)',
        },
        point: {
          type: 'string',
          description: 'Point prompt as "x,y" (default "486,301")',
        },
        box: {
          type: 'string',
          description: 'Box prompt as "x1,y1,x2,y2" (default "412,109,717,449")',
        },
        language: {
          type: 'string',
          description: 'Text prompt for SAM3 (default "girl")',
        },
      },
      required: ['video_dir'],
    },
    output: {
      schema: { type: 'string' },
      render: (_args, value) => [{ type: 'text', text: value }],
    },
    async execute(args, exec) {
      const script = args.model === 'sam3' ? 'interactive_sam3.py' : 'interactive_sam2.py'
      const argv = []
      if (script === 'interactive_sam2.py') {
        argv.push('--video_dir', args.video_dir, '--output_dir', args.output_dir ?? 'output_video')
      } else {
        argv.push('--video_dir', args.video_dir)
        if (args.output_dir) argv.push('--output_dir', args.output_dir)
      }
      if (args.prompt_type) argv.push('--prompt_type', args.prompt_type)
      if (args.prompt_type === 'language' && args.language) {
        argv.push('--language', args.language)
      } else if (args.prompt_type === 'box' && args.box) {
        argv.push('--bbox', ...args.box.split(','))
      } else if (args.point) {
        argv.push('--point', ...args.point.split(','))
      }
      return runPython(exec, script, argv)
    },
  })
}
