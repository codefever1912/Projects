import React from 'react';

const PitchMap = ({ players = [], ball = null, stats = { home_poss: 0, away_poss: 0 } }) => {
  // Cyberpunk Color Palette Constants
  const COLOR_HOME = '#39ff14'; // Neon Green
  const COLOR_AWAY = '#00f3ff'; // Electric Blue
  const COLOR_BALL = '#ffff00'; // Neon Yellow

  // Calculate Possession Percentages
  const totalFrames = (stats.home_poss + stats.away_poss) || 1; // Avoid div by zero
  const homePct = Math.round((stats.home_poss / totalFrames) * 100);
  const awayPct = 100 - homePct;

  return (
    <div className="flex flex-col gap-4 w-full max-w-md mx-auto p-6 bg-slate-900 rounded-xl border border-slate-700 shadow-2xl font-mono">
      
      {/* Header */}
      <div className="flex justify-between items-end border-b border-slate-700 pb-2">
        <h2 className="text-xl font-bold tracking-widest text-white uppercase drop-shadow-[0_0_5px_rgba(255,255,255,0.5)]">
          PITCH<span className="text-[#39ff14]">VISION</span>
        </h2>
        <div className="flex gap-2 text-xs">
          <div className="flex items-center gap-1">
            <div className="w-2 h-2 rounded-full bg-[#39ff14] shadow-[0_0_5px_#39ff14]"></div>
            <span className="text-slate-300">HOME</span>
          </div>
          <div className="flex items-center gap-1">
            <div className="w-2 h-2 rounded-full bg-[#00f3ff] shadow-[0_0_5px_#00f3ff]"></div>
            <span className="text-slate-300">AWAY</span>
          </div>
        </div>
      </div>

      {/* The Pitch - Aspect Ratio 2:3 (Vertical) */}
      <div className="relative w-full aspect-[2/3] bg-slate-950 border-2 border-[#39ff14] rounded overflow-hidden shadow-[0_0_20px_rgba(57,255,20,0.15)]">
        
        {/* Pitch Markings: Center Line */}
        <div className="absolute top-1/2 left-0 w-full h-[1px] bg-[#39ff14] opacity-40"></div>
        
        {/* Pitch Markings: Center Circle */}
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-24 h-24 border border-[#39ff14] rounded-full opacity-40"></div>
        
        {/* Pitch Markings: Goals (Visual Only) */}
        <div className="absolute top-0 left-1/2 -translate-x-1/2 w-1/3 h-8 border-b border-x border-[#39ff14] opacity-30"></div>
        <div className="absolute bottom-0 left-1/2 -translate-x-1/2 w-1/3 h-8 border-t border-x border-[#39ff14] opacity-30"></div>

        {/* Render Players */}
        {players.map((p, index) => {
          const teamColor = p.team === 0 ? COLOR_HOME : COLOR_AWAY;
          return (
            <div
              key={index}
              className="absolute w-3 h-3 rounded-full -translate-x-1/2 -translate-y-1/2 transition-all duration-75 ease-linear"
              style={{
                left: `${p.x * 100}%`,
                top: `${p.y * 100}%`,
                backgroundColor: teamColor,
                boxShadow: `0 0 8px ${teamColor}`
              }}
            />
          );
        })}

        {/* Render Ball */}
        {ball && (
          <div
            className="absolute w-2.5 h-2.5 bg-yellow-400 rounded-full -translate-x-1/2 -translate-y-1/2 transition-all duration-75 ease-linear z-10 border border-black"
            style={{
              left: `${ball.x * 100}%`,
              top: `${ball.y * 100}%`,
              boxShadow: `0 0 10px ${COLOR_BALL}`
            }}
          />
        )}
      </div>

      {/* Possession Power Bar */}
      <div className="w-full">
        <div className="flex justify-between text-xs mb-1 font-bold">
          <span style={{ color: COLOR_HOME }}>HOME POSS: {homePct}%</span>
          <span style={{ color: COLOR_AWAY }}>AWAY POSS: {awayPct}%</span>
        </div>
        
        <div className="h-4 w-full flex bg-slate-800 rounded overflow-hidden border border-slate-600">
          {/* Home Bar */}
          <div
            className="h-full transition-all duration-300 flex items-center justify-center"
            style={{
              width: `${homePct}%`,
              backgroundColor: COLOR_HOME,
              boxShadow: `inset 0 0 10px rgba(0,0,0,0.3)`
            }}
          >
          </div>
          
          {/* Away Bar */}
          <div
            className="h-full transition-all duration-300 flex items-center justify-center"
            style={{
              width: `${awayPct}%`,
              backgroundColor: COLOR_AWAY,
              boxShadow: `inset 0 0 10px rgba(0,0,0,0.3)`
            }}
          >
          </div>
        </div>
      </div>

    </div>
  );
};

export default PitchMap;