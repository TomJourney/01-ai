interface StatsBarProps {
  activeCount: number;
  doneCount: number;
  onClearDone: () => void;
}

export default function StatsBar({ activeCount, doneCount, onClearDone }: StatsBarProps) {
  return (
    <div className="stats">
      <span>未完成 {activeCount} 项，已完成 {doneCount} 项</span>
      {doneCount > 0 && (
        <button className="clear-btn" onClick={onClearDone}>清除已完成</button>
      )}
    </div>
  );
}
