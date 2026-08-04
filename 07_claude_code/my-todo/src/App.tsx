import { useState } from 'react';
import type { FilterType } from './types';
import { useTodos } from './hooks/useTodos';
import TodoInput from './components/TodoInput';
import FilterBar from './components/FilterBar';
import TodoList from './components/TodoList';
import StatsBar from './components/StatsBar';
import './App.css';

export default function App() {
  const { todos, addTodo, toggleTodo, deleteTodo, clearDone } = useTodos();
  const [filter, setFilter] = useState<FilterType>('all');

  const activeCount = todos.filter(t => !t.done).length;
  const doneCount = todos.length - activeCount;

  return (
    <div className="container">
      <h1>待办事项</h1>
      <TodoInput onAdd={addTodo} />
      <FilterBar currentFilter={filter} onFilterChange={setFilter} />
      <div className="card">
        <StatsBar activeCount={activeCount} doneCount={doneCount} onClearDone={clearDone} />
        <TodoList
          todos={todos}
          filter={filter}
          onToggle={toggleTodo}
          onDelete={deleteTodo}
        />
      </div>
    </div>
  );
}
