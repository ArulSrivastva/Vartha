import { useEffect, useState } from 'react';
import Header from './components/Header';
import LiveCycloneStatus from './components/LiveCycloneStatus';
import UnifiedAnalysisPanel from './components/UnifiedAnalysisPanel';
import MapView from './components/map/MapView';
import SatelliteViewer from './components/SatelliteViewer';
import ForecastCard from './components/ForecastCard';
import ForecastMotionCard from './components/ForecastMotionCard';
import LandfallPanel from './components/LandfallPanel';
import RiskIndicator from './components/RiskIndicator';
import ChartsPanel from './components/charts/ChartsPanel';
import ModelIntelligence from './components/ModelIntelligence';
import BaselineComparison from './components/BaselineComparison';
import PipelineStatus from './components/PipelineStatus';
import ProvenancePanel from './components/ProvenancePanel';
import HistoryEditor from './components/HistoryEditor';
import ExportReportModal from './components/ExportReportModal';
import { fetchAnalyze } from './api/client';

export default function App() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [imageResult, setImageResult] = useState(null);
  const [backendOffline, setBackendOffline] = useState(false);
  const [isExportOpen, setIsExportOpen] = useState(false);

  useEffect(() => {
    let cancelled = false;
    fetchAnalyze()
      .then(res => { 
        if(!cancelled){ 
          setData(res); 
          setLoading(false); 
          setBackendOffline(false);
        } 
      })
      .catch(err => { 
        if(!cancelled){ 
          setError(err.message); 
          setLoading(false); 
          setBackendOffline(true);
        } 
      });
    return () => { cancelled = true; };
  }, []);

  function handleHistoryAnalyze(history){
    setLoading(true);
    setError(null);
    fetchAnalyze(history)
      .then(res => { 
        setData(res); 
        setLoading(false); 
        setBackendOffline(false);
      })
      .catch(err => { 
        setError(err.message); 
        setLoading(false); 
        setBackendOffline(true);
      });
  }

  function handleReanalyze(){
    handleHistoryAnalyze();
  }

  function handleImageResult(result){
    setImageResult(result);
    setData(prev => {
      if (!prev) return result;
      return {
        ...prev,
        meta: {
          ...prev.meta,
          source: result.meta?.source || result.sourceLabel || "User Upload",
        },
        detection: {
          ...prev.detection,
          ...result.detection,
          detected: result.detection?.detected !== undefined ? result.detection.detected : prev.detection?.detected,
          confidence: result.detection?.confidence !== undefined ? result.detection.confidence : prev.detection?.confidence,
          location: result.detection?.location || prev.detection?.location,
          movementDirection: result.detection?.movementDirection || prev.detection?.movementDirection,
          movementSpeedKmh: result.detection?.movementSpeedKmh || prev.detection?.movementSpeedKmh,
          structuralPattern: result.detection?.structuralPattern || prev.detection?.structuralPattern,
        },
        classification: {
          ...prev.classification,
          ...result.classification,
          category: result.classification?.category || prev.classification?.category,
          windSpeedKmh: result.classification?.windSpeedKmh ?? prev.classification?.windSpeedKmh,
          pressureHpa: result.classification?.pressureHpa ?? prev.classification?.pressureHpa,
          confidence: result.classification?.confidence ?? prev.classification?.confidence,
        },
        satellite: {
          ...prev.satellite,
          ...result.satellite,
          boundingBox: result.satellite?.boundingBox || prev.satellite?.boundingBox,
        },
        provenance: result.provenance || prev.provenance,
      };
    });
  }

  return (
    <div className="min-h-screen">
      <Header meta={data?.meta} loading={loading} onExport={() => setIsExportOpen(true)} />

      <main className="max-w-[1400px] mx-auto px-6 py-6 space-y-6">
        
        {/* Error notification */}
        {error && (
          <div className="text-[13px] text-risk-high bg-risk-high/10 border border-risk-high/25 rounded-xl px-4 py-3">
            Could not load analysis data: {error}
          </div>
        )}

        {/* Live system status */}
        <LiveCycloneStatus data={data} loading={loading} />

        {/* Unified analysis panel */}
        <UnifiedAnalysisPanel
          data={data}
          loading={loading}
          onReanalyze={handleReanalyze}
          backendOffline={backendOffline}
        />

        {/* Dashboard main grid */}
        <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
          
          {/* Spatial tracking column */}
          <div className="xl:col-span-2 space-y-6">
            <MapView data={data} loading={loading} />
            <SatelliteViewer
              satellite={data?.satellite}
              meta={data?.meta}
              loading={loading}
              onImageResult={handleImageResult}
            />
            <HistoryEditor onAnalyze={handleHistoryAnalyze} />
            <PipelineStatus loading={loading} />
            <ChartsPanel data={data} loading={loading} />
          </div>

          {/* Forecast sidebar */}
          <div className="space-y-6">
            <ForecastCard forecast={data?.forecast} loading={loading} />
            <ForecastMotionCard detection={data?.detection} forecast={data?.forecast} loading={loading} />
            <LandfallPanel landfall={data?.landfall} loading={loading} />
            <RiskIndicator risk={data?.risk}
                           classification={data?.classification}
                           landfall={data?.landfall} loading={loading} />
            <ModelIntelligence />
            <BaselineComparison />
          </div>

        </div>

        {/* Model provenance disclosures */}
        <ProvenancePanel provenance={data?.provenance} loading={loading} />

        {/* User image provenance */}
        {imageResult && (
          <ProvenancePanel
            title="User Image Analysis Provenance"
            provenance={imageResult?.provenance}
            loading={false}
          />
        )}

        <ExportReportModal
          data={data}
          isOpen={isExportOpen}
          onClose={() => setIsExportOpen(false)}
        />

      </main>

      <footer className="max-w-[1400px] mx-auto px-6 py-8 text-[11.5px] text-ink-faint border-t border-border mt-8">
        <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
          <span>
            VARTHA Cyclone Intelligence System · Experimental research prototype for SIH 2026. Real-time inference across satellite vision, multi-source atmospheric classification, and recurrent trajectory forecasting.
          </span>
          <span className="font-mono text-[10px] text-accent-strong bg-accent-soft/30 px-2 py-0.5 rounded">
            SIH 2026 Forecaster UI Handoff
          </span>
        </div>
      </footer>
    </div>
  );
}
