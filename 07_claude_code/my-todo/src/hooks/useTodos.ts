import { useState, useCallback } from 'react';
import type { Todo } from '../types';

const STORAGE_KEY = 'my-todo-app-data';

function loadTodos(): Todo[] {
  try {
    return JSON.parse(localStorage.getItem(STORAGE_KEY) || '[]');
  } catch {
    return [];
  }
}

function saveTodos(todos: Todo[]) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(todos));
}

export function useTodos() {
  const [todos, setTodos] = useState<Todo[]>(loadTodos);

  const addTodo = useCallback((text: string) => {
    setTodos(prev => [{ id: Date.now(), text, done: false }, ...prev]);
    setTodos(prev => { saveTodos(prev); return prev; });
  }, []);

  const toggleTodo = useCallback((id: number) => {
    setTodos(prev => {
      const next = prev.map(t => t.id === id ? { ...t, done: !t.done } : t);
      saveTodos(next);
      return next;
    });
  }, []);

  const deleteTodo = useCallback((id: number) => {
    setTodos(prev => {
      const next = prev.filter(t => t.id !== id);
      saveTodos(next);
      return next;
    });
  }, []);

  const clearDone = useCallback(() => {
    setTodos(prev => {
      const next = prev.filter(t => !t.done);
      saveTodos(next);
      return next;
    });
  }, []);

  return { todos, addTodo, toggleTodo, deleteTodo, clearDone };
}
