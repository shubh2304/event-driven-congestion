'use client';

import { useState, useEffect, useMemo } from 'react';
import dynamic from 'next/dynamic';
import { getHotspots } from '@/lib/api';
import PageTransition from '@/components/PageTransition';
import { IconMap, IconMapPin } from '@/components/SvgIcons';

const CLUSTER_COLORS = [
  '#ef4444', '#3b82f6', '#10b981', '#f59e0b', '#8b5cf6',
  '#ec4899', '#14b8a6', '#f97316', '#6366f1', '#84cc16',
  '#06b6d4', '#e11d48', '#a855f7', '#22c55e', '#eab308'
];

// Single dynamic import for the entire map component to avoid
// react-leaflet context sharing issues when importing subcomponents separately.
const HotspotMap = dynamic(
  () => import('react-leaflet').then((mod) => {
    const { MapContainer, TileLayer, CircleMarker, Popup } = mod;

    function MapInner({ points }) {
      return (
        <MapContainer
          center={[12.9716, 77.5946]}
          zoom={12}
          style={{ width: '100%', height: '100%' }}
          scrollWheelZoom={true}
        >
          <TileLayer
            url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"
            attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OSM</a> &copy; <a href="https://carto.com/">CARTO</a>'
          />
          {points.map((pt, i) => (
            <CircleMarker
              key={i}
              center={[pt.latitude, pt.longitude]}
              radius={5}
              fillColor={CLUSTER_COLORS[pt.cluster % CLUSTER_COLORS.length]}
              color={CLUSTER_COLORS[pt.cluster % CLUSTER_COLORS.length]}
              weight={1}
              opacity={0.8}
              fillOpacity={0.6}
            >
              <Popup>
                <div style={{ fontFamily: 'Outfit, sans-serif', fontSize: '0.85rem' }}>
                  <strong>Cluster {pt.cluster}</strong><br />
                  Cause: {pt.event_cause}<br />
                  Priority: {pt.priority}
                </div>
              </Popup>
            </CircleMarker>
          ))}
        </MapContainer>
      );
    }

    return MapInner;
  }),
  {
    ssr: false,
    loading: () => (
      <div style={{ width: '100%', height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', background: 'var(--bg-surface)' }}>
        <div className="loading-spinner" />
      </div>
    ),
  }
);

export default function HotspotsPage() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
    getHotspots()
      .then(setData)
      .catch(err => setError(err.message))
      .finally(() => setLoading(false));
  }, []);

  // Load leaflet CSS once on mount
  useEffect(() => {
    if (mounted) {
      // Only add if not already present
      if (!document.querySelector('link[href*="leaflet"]')) {
        const link = document.createElement('link');
        link.rel = 'stylesheet';
        link.href = 'https://unpkg.com/leaflet@1.9.4/dist/leaflet.css';
        document.head.appendChild(link);
      }
    }
  }, [mounted]);

  const sortedProfiles = useMemo(() => {
    if (!data?.profiles) return [];
    return [...data.profiles].sort((a, b) => b.event_count - a.event_count);
  }, [data]);

  if (loading) return <PageTransition><div className="loading-spinner" /></PageTransition>;

  if (error) return (
    <PageTransition>
      <div className="page-header">
        <h1 className="page-header__title">Hotspot Map</h1>
      </div>
      <div style={{ padding: '2rem', color: 'var(--danger)' }}>{error}</div>
    </PageTransition>
  );

  return (
    <PageTransition>
      <div className="page-header">
        <h1 className="page-header__title">Bengaluru Event Hotspot Map</h1>
        <p className="page-header__desc">
          Geospatial visualization of DBSCAN-identified congestion hotspot clusters across Bengaluru.
        </p>
      </div>

      {mounted && data && (
        <div className="map-container" style={{ marginBottom: '1.5rem' }}>
          <HotspotMap points={data.points} />
        </div>
      )}

      <h2 className="section-title">
        <IconMapPin size={20} />
        Cluster Profiles
      </h2>
      <p style={{ color: 'var(--text-secondary)', fontSize: '0.9rem', marginBottom: '1rem' }}>
        Metrics for each density-based cluster (centroids, sizes, high priority rate, and dominant cause).
      </p>

      <div className="card" style={{ overflow: 'auto' }}>
        <table className="data-table" id="cluster-profiles-table">
          <thead>
            <tr>
              <th>Cluster ID</th>
              <th>Total Events</th>
              <th>High Priority (%)</th>
              <th>Centroid Lat</th>
              <th>Centroid Lon</th>
              <th>Top Cause</th>
            </tr>
          </thead>
          <tbody>
            {sortedProfiles.map((row, i) => (
              <tr key={i}>
                <td>
                  <span style={{
                    display: 'inline-block', width: 10, height: 10, borderRadius: '50%',
                    background: CLUSTER_COLORS[row.cluster % CLUSTER_COLORS.length],
                    marginRight: 8, verticalAlign: 'middle'
                  }} />
                  {row.cluster}
                </td>
                <td>{row.event_count}</td>
                <td>{row.high_priority_pct?.toFixed(1)}%</td>
                <td>{row.centroid_lat?.toFixed(4)}</td>
                <td>{row.centroid_lon?.toFixed(4)}</td>
                <td>{row.top_cause}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </PageTransition>
  );
}

