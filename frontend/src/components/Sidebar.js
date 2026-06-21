'use client';

import { useState, useEffect } from 'react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { IconPredictor, IconMap, IconChart, IconPerformance, IconChat, IconMonitor, IconTraffic } from './SvgIcons';

const navItems = [
  { href: '/', label: 'Event Predictor', icon: IconPredictor },
  { href: '/hotspots', label: 'Hotspot Map', icon: IconMap },
  { href: '/analytics', label: 'Dataset Analytics', icon: IconChart },
  { href: '/performance', label: 'Model Performance', icon: IconPerformance },
  { href: '/assistant', label: 'ASTRAM Assistant', icon: IconChat },
  { href: '/monitoring', label: 'Model Monitoring', icon: IconMonitor },
];

export default function Sidebar() {
  const pathname = usePathname();
  const [mobileOpen, setMobileOpen] = useState(false);

  // Close sidebar on route change
  useEffect(() => {
    setMobileOpen(false);
  }, [pathname]);

  // Prevent body scroll when mobile sidebar is open
  useEffect(() => {
    if (mobileOpen) {
      document.body.style.overflow = 'hidden';
    } else {
      document.body.style.overflow = '';
    }
    return () => { document.body.style.overflow = ''; };
  }, [mobileOpen]);

  return (
    <>
      {/* Mobile hamburger button */}
      <button
        className="mobile-menu-btn"
        onClick={() => setMobileOpen(prev => !prev)}
        aria-label="Toggle navigation menu"
        id="mobile-menu-toggle"
      >
        <span className={`hamburger ${mobileOpen ? 'hamburger--active' : ''}`}>
          <span />
          <span />
          <span />
        </span>
      </button>

      {/* Overlay for mobile */}
      {mobileOpen && (
        <div
          className="sidebar-overlay"
          onClick={() => setMobileOpen(false)}
          aria-hidden="true"
        />
      )}

      <aside className={`sidebar ${mobileOpen ? 'sidebar--open' : ''}`} id="main-sidebar">
        <div className="sidebar__brand">
          <div className="sidebar__logo">
            <IconTraffic size={28} />
          </div>
          <div className="sidebar__brand-text">
            <h1 className="sidebar__title">ASTRAM</h1>
            <p className="sidebar__subtitle">Congestion Forecaster</p>
          </div>
        </div>

        <nav className="sidebar__nav" id="main-navigation">
          {navItems.map((item) => {
            const isActive = pathname === item.href;
            const Icon = item.icon;
            return (
              <Link
                key={item.href}
                href={item.href}
                className={`sidebar__link ${isActive ? 'sidebar__link--active' : ''}`}
                id={`nav-${item.href.replace('/', '') || 'home'}`}
              >
                <span className="sidebar__link-icon">
                  <Icon size={20} />
                </span>
                <span className="sidebar__link-label">{item.label}</span>
                {isActive && <span className="sidebar__link-indicator" />}
              </Link>
            );
          })}
        </nav>

        <div className="sidebar__footer">
          <div className="sidebar__info-card">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <circle cx="12" cy="12" r="10" />
              <line x1="12" y1="16" x2="12" y2="12" />
              <line x1="12" y1="8" x2="12.01" y2="8" />
            </svg>
            <div>
              <strong>Bengaluru Traffic Management</strong>
              <p>Using the ASTRAM dataset to run live predictions and recommend optimal field resources.</p>
            </div>
          </div>
        </div>
      </aside>
    </>
  );
}
