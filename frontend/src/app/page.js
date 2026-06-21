'use client';

import { useState } from 'react';
import { predict } from '@/lib/api';
import PageTransition from '@/components/PageTransition';
import MetricCard from '@/components/MetricCard';
import GaugeChart from '@/components/GaugeChart';
import Badge from '@/components/Badge';
import { IconShield, IconRoadblock, IconClock, IconMapPin, IconUsers, IconCone, IconSplit, IconZap } from '@/components/SvgIcons';

const EVENT_CAUSES = [
  'vehicle_breakdown', 'accident', 'construction', 'pot_holes',
  'water_logging', 'public_event', 'procession', 'vip_movement',
  'protest', 'tree_fall', 'road_conditions', 'congestion',
  'fog_low_visibility', 'others'
];

const VEHICLE_TYPES = [
  'others', 'bmtc_bus', 'heavy_vehicle', 'lcv', 'truck', 'private_bus',
  'private_car', 'ksrtc_bus', 'taxi', 'auto'
];

const ZONES = [
  'Central Zone 2', 'West Zone 1', 'North Zone 2', 'West Zone 2',
  'South Zone 2', 'North Zone 1', 'Central Zone 1', 'East Zone 1',
  'South Zone 1', 'East Zone 2'
];

const CORRIDORS = [
  'Non-corridor', 'Mysore Road', 'Bellary Road 1', 'Tumkur Road',
  'Bellary Road 2', 'Hosur Road', 'ORR North 1', 'Old Madras Road',
  'Magadi Road', 'ORR East 1', 'Other'
];

const DAYS = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'];

export default function PredictorPage() {
  const [form, setForm] = useState({
    event_type: 'unplanned',
    event_cause: 'vehicle_breakdown',
    veh_type: 'others',
    zone: 'Central Zone 2',
    corridor: 'Non-corridor',
    police_station: 'Cubbon Park',
    latitude: 12.9716,
    longitude: 77.5946,
    hour: 8,
    day_of_week: 0,
    month: 6,
  });

  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleSubmit = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await predict(form);
      setResult(res);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleChange = (field, value) => {
    setForm(prev => ({ ...prev, [field]: value }));
  };

  return (
    <PageTransition>
      <div className="page-header">
        <h1 className="page-header__title">Event Impact Predictor</h1>
        <p className="page-header__desc">
          Forecast traffic congestion impact in real time and get actionable deployment recommendations.
        </p>
      </div>

      <div className="grid-3" style={{ marginBottom: '1.5rem' }}>
        {/* Column 1: Event Details */}
        <div className="card">
          <div className="card__title">
            <IconZap size={18} />
            Event Details
          </div>
          <div className="form-group">
            <label className="form-label">Event Type</label>
            <select className="form-select" value={form.event_type} onChange={e => handleChange('event_type', e.target.value)} id="event-type">
              <option value="unplanned">Unplanned</option>
              <option value="planned">Planned</option>
            </select>
          </div>
          <div className="form-group">
            <label className="form-label">Event Cause</label>
            <select className="form-select" value={form.event_cause} onChange={e => handleChange('event_cause', e.target.value)} id="event-cause">
              {EVENT_CAUSES.map(c => <option key={c} value={c}>{c.replace(/_/g, ' ')}</option>)}
            </select>
          </div>
          <div className="form-group">
            <label className="form-label">Vehicle Type</label>
            <select className="form-select" value={form.veh_type} onChange={e => handleChange('veh_type', e.target.value)} id="vehicle-type">
              {VEHICLE_TYPES.map(v => <option key={v} value={v}>{v.replace(/_/g, ' ')}</option>)}
            </select>
          </div>
        </div>

        {/* Column 2: Geospatial Context */}
        <div className="card">
          <div className="card__title">
            <IconMapPin size={18} />
            Geospatial Context
          </div>
          <div className="form-group">
            <label className="form-label">Bengaluru Zone</label>
            <select className="form-select" value={form.zone} onChange={e => handleChange('zone', e.target.value)} id="zone">
              {ZONES.map(z => <option key={z} value={z}>{z}</option>)}
            </select>
          </div>
          <div className="form-group">
            <label className="form-label">Corridor</label>
            <select className="form-select" value={form.corridor} onChange={e => handleChange('corridor', e.target.value)} id="corridor">
              {CORRIDORS.map(c => <option key={c} value={c}>{c}</option>)}
            </select>
          </div>
          <div className="form-group">
            <label className="form-label">Police Station</label>
            <input className="form-input" type="text" value={form.police_station} onChange={e => handleChange('police_station', e.target.value)} id="police-station" />
          </div>
        </div>

        {/* Column 3: Location & Timing */}
        <div className="card">
          <div className="card__title">
            <IconClock size={18} />
            Location & Timing
          </div>
          <div className="form-group">
            <label className="form-label">Latitude</label>
            <input className="form-input" type="number" value={form.latitude} onChange={e => handleChange('latitude', parseFloat(e.target.value))} step="0.0001" min="12.75" max="13.30" id="latitude" />
          </div>
          <div className="form-group">
            <label className="form-label">Longitude</label>
            <input className="form-input" type="number" value={form.longitude} onChange={e => handleChange('longitude', parseFloat(e.target.value))} step="0.0001" min="77.25" max="77.85" id="longitude" />
          </div>
          <div className="form-group">
            <label className="form-label">Hour of Day: <span className="slider-value">{form.hour}:00</span></label>
            <input className="form-slider" type="range" min="0" max="23" value={form.hour} onChange={e => handleChange('hour', parseInt(e.target.value))} id="hour-slider" />
          </div>
          <div className="form-group">
            <label className="form-label">Day of Week</label>
            <select className="form-select" value={form.day_of_week} onChange={e => handleChange('day_of_week', parseInt(e.target.value))} id="day-of-week">
              {DAYS.map((d, i) => <option key={d} value={i}>{d}</option>)}
            </select>
          </div>
          <div className="form-group">
            <label className="form-label">Month: <span className="slider-value">{form.month}</span></label>
            <input className="form-slider" type="range" min="1" max="12" value={form.month} onChange={e => handleChange('month', parseInt(e.target.value))} id="month-slider" />
          </div>
        </div>
      </div>

      <button className="btn btn--primary btn--full btn--lg" onClick={handleSubmit} disabled={loading} id="predict-btn">
        {loading ? (
          <><span className="loading-spinner" style={{ width: 18, height: 18, margin: 0, borderWidth: 2 }} /> Running Models...</>
        ) : (
          <><IconShield size={20} /> Predict & Recommend</>
        )}
      </button>

      {error && (
        <div style={{ marginTop: '1rem', padding: '1rem', background: 'var(--danger-soft)', border: '1px solid var(--danger)', borderRadius: 'var(--radius-sm)', color: 'var(--danger)' }}>
          {error}
        </div>
      )}

      {result && (
        <div style={{ marginTop: '2rem' }}>
          <hr className="divider" />
          <h2 className="section-title">
            <IconShield size={20} />
            Model Predictions & Impact Score
          </h2>

          <div className="metrics-grid"  style={{ gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))' }}>
            <MetricCard
              icon={<IconShield size={18} />}
              title="Priority Risk"
              value={`${(result.priority_risk * 100).toFixed(1)}%`}
              badge={{ label: result.priority_label, type: result.priority_label === 'HIGH' ? 'high' : 'low' }}
              delay={0}
            />
            <MetricCard
              icon={<IconRoadblock size={18} />}
              title="Road Closure Risk"
              value={`${(result.closure_risk * 100).toFixed(1)}%`}
              badge={{ label: result.closure_label, type: result.closure_label === 'LIKELY' ? 'high' : 'low' }}
              delay={100}
            />
            <MetricCard
              icon={<IconClock size={18} />}
              title="Est. Duration"
              value={`${result.estimated_duration_min} min`}
              subtitle="Log-transformed LGBM Regressor"
              delay={200}
            />
            <MetricCard
              icon={<IconMapPin size={18} />}
              title="Context Zone"
              value={form.zone}
              subtitle={`Geohash: ${result.geohash}`}
              delay={300}
            />
          </div>

          {/* Gauge */}
          <div className="card" style={{ textAlign: 'center', marginBottom: '1.5rem' }}>
            <GaugeChart value={result.priority_risk * 100} />
          </div>

          {/* Recommendations */}
          <h2 className="section-title">
            <IconUsers size={20} />
            Operational Recommendations
          </h2>
          <div className="rec-grid">
            <div className="rec-card rec-card--info" style={{ animationDelay: '0.1s' }}>
              <div className="rec-card__header">
                <IconUsers size={18} style={{ color: 'var(--info)' }} />
                Manpower
              </div>
              <div className="rec-card__body">{result.recommendations.manpower}</div>
            </div>
            <div className="rec-card rec-card--warning" style={{ animationDelay: '0.2s' }}>
              <div className="rec-card__header">
                <IconCone size={18} style={{ color: 'var(--warning)' }} />
                Barricading
              </div>
              <div className="rec-card__body">{result.recommendations.barricading}</div>
            </div>
            <div className="rec-card rec-card--danger" style={{ animationDelay: '0.3s' }}>
              <div className="rec-card__header">
                <IconSplit size={18} style={{ color: 'var(--danger)' }} />
                Diversion Urgency
              </div>
              <div className="rec-card__body">{result.recommendations.diversion}</div>
            </div>
          </div>
        </div>
      )}
    </PageTransition>
  );
}
