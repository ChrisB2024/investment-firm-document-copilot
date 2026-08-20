/**
 * The only module allowed to read `import.meta.env`.
 *
 * Everything else imports `env` from here. Validation runs once at module load,
 * so a missing var fails the app immediately instead of surfacing as
 * `undefined` in a fetch URL three screens later.
 */

export type Env = {
  apiBaseUrl: string
  supabaseUrl: string
  supabaseAnonKey: string
}

// The three accesses below are written out literally on purpose: Vite replaces
// `import.meta.env.VITE_X` at build time, so indexing it dynamically yields
// undefined in the production bundle.
function loadEnv(): Env {
  const missing: string[] = []

  const req = (name: string, value: unknown): string => {
    const trimmed = typeof value === 'string' ? value.trim() : ''
    if (trimmed !== '') return trimmed
    missing.push(name)
    return ''
  }

  const parsed: Env = {
    apiBaseUrl: req('VITE_API_BASE_URL', import.meta.env.VITE_API_BASE_URL),
    supabaseUrl: req('VITE_SUPABASE_URL', import.meta.env.VITE_SUPABASE_URL),
    supabaseAnonKey: req('VITE_SUPABASE_ANON_KEY', import.meta.env.VITE_SUPABASE_ANON_KEY),
  }

  if (missing.length > 0) {
    throw new Error(
      `Missing or empty environment variable(s): ${missing.join(', ')}. ` +
        `Add them to frontend/.env (see .env.example) and restart — Vite only ` +
        `reads .env files at startup.`
    )
  }

  return parsed
}

export const env: Env = loadEnv()
