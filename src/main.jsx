import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.jsx'
import { HelmetProvider, Helmet } from "react-helmet-async";
import { Analytics } from "@vercel/analytics/react"


createRoot(document.getElementById("root")).render(
  <HelmetProvider>
    <App />
  </HelmetProvider>
);