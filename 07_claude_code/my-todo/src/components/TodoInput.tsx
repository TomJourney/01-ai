import { useRef } from 'react';

interface TodoInputProps {
  onAdd: (text: string) => void;
}

export default function TodoInput({ onAdd }: TodoInputProps) {
  const inputRef = useRef<HTMLInputElement>(null);

  const handleSubmit = () => {
    const text = inputRef.current?.value.trim() || '';
    if (!text) return;
    onAdd(text);
    inputRef.current!.value = '';
  };

  return (
    <div className="input-row">
      <input
        ref={inputRef}
        type="text"
        placeholder="输入新的待办事项..."
        onKeyDown={e => { if (e.key === 'Enter') handleSubmit(); }}
      />
      <button onClick={handleSubmit}>添加</button>
    </div>
  );
}
