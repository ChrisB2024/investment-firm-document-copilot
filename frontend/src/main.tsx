import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
// Imported for its side effect: validates config before anything renders, so a
// missing var is a blank page with a clear console error, not a broken fetch later.
import '@/lib/env'
import App from './App.tsx'

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
