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

// TODO: read the three VITE_ vars and throw if any is missing or empty.
//
// Two things worth getting right:
//   1. Report ALL missing vars in one error, not just the first — otherwise
//      setting up a new machine is a game of whack-a-mole.
//   2. Vite statically replaces `import.meta.env.VITE_X` at build time, so you
//      must write the property access literally. Looping over a list of names
//      and indexing dynamically will silently produce undefined in the build.
function loadEnv(): Env {
  throw new Error('TODO: implement loadEnv')
}

export const env: Env = loadEnv()
