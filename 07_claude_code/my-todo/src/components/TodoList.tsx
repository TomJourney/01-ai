import type { Todo, FilterType } from '../types';
import TodoItem from './TodoItem';

interface TodoListProps {
  todos: Todo[];
  filter: FilterType;
  onToggle: (id: number) => void;
  onDelete: (id: number) => void;
}

export default function TodoList({ todos, filter, onToggle, onDelete }: TodoListProps) {
  const filtered = todos.filter(t => {
    if (filter === 'active') return !t.done;
    if (filter === 'done') return t.done;
    return true;
  });

  if (filtered.length === 0) {
    return (
      <div className="empty">
        {todos.length === 0 ? '暂无待办事项，开始添加吧！' : '没有符合条件的待办事项'}
      </div>
    );
  }

  return (
    <ul className="todo-list">
      {filtered.map(todo => (
        <TodoItem
          key={todo.id}
          todo={todo}
          onToggle={onToggle}
          onDelete={onDelete}
        />
      ))}
    </ul>
  );
}
