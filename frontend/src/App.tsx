import { BrowserRouter, Routes, Route } from 'react-router-dom'

/**
 * Router shell.
 *
 * TODO (Phase 6): add the real routes — /login, /chat, /chat/:threadId — and a
 * protected-route wrapper that redirects to /login when there is no Supabase
 * session. For now this only proves the toolchain works end to end.
 */
export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route
          path="/"
          element={
            <main className="grid min-h-screen place-items-center">
              <h1 className="text-2xl font-semibold">Document Copilot</h1>
            </main>
          }
        />
      </Routes>
    </BrowserRouter>
  )
}
