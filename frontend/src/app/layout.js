import './globals.css';
import Sidebar from '@/components/Sidebar';

export const metadata = {
  title: 'ASTRAM — Bengaluru Event Congestion Forecaster',
  description: 'ML-powered traffic congestion prediction, hotspot analysis, and resource recommendation system for Bengaluru traffic management.',
};

export default function RootLayout({ children }) {
  return (
    <html lang="en" data-scroll-behavior="smooth" suppressHydrationWarning>
      <head>
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <meta name="theme-color" content="#07070d" />
      </head>
      <body suppressHydrationWarning>
        <div className="app-layout">
          <Sidebar />
          <main className="main-content">
            {children}
          </main>
        </div>
      </body>
    </html>
  );
}
