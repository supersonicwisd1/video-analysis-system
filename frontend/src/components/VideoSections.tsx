import React, { useState, useEffect } from 'react';
import { Clock, ChevronDown, ChevronUp, Edit2, Trash2, Plus } from 'lucide-react';
import { VideoSection } from '../types/api';

interface VideoSectionsProps {
  videoId: string;
  sections: VideoSection[];
  currentTime: number;
  onSeek: (timestamp: number) => void;
  onSectionUpdate?: (section: VideoSection) => Promise<void>;
  onSectionDelete?: (sectionId: string) => Promise<void>;
  onSectionCreate?: (section: Omit<VideoSection, 'id'>) => Promise<void>;
  isEditing?: boolean;
}

export const VideoSections: React.FC<VideoSectionsProps> = ({
  videoId,
  sections,
  currentTime,
  onSeek,
  onSectionUpdate,
  onSectionDelete,
  onSectionCreate,
  isEditing = false
}) => {
  const [expandedSections, setExpandedSections] = useState<Set<string>>(new Set());
  const [editingSection, setEditingSection] = useState<string | null>(null);
  const [newSection, setNewSection] = useState<Partial<VideoSection>>({
    title: '',
    start_time: 0,
    end_time: 0,
    type: 'manual'
  });

  // Find current section
  const currentSection = sections.find(
    section => currentTime >= section.start_time && currentTime <= section.end_time
  );

  // Format time
  const formatTime = (seconds: number) => {
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = Math.floor(seconds % 60);
    return `${minutes}:${remainingSeconds.toString().padStart(2, '0')}`;
  };

  // Toggle section expansion
  const toggleSection = (sectionId: string) => {
    setExpandedSections(prev => {
      const next = new Set(prev);
      if (next.has(sectionId)) {
        next.delete(sectionId);
      } else {
        next.add(sectionId);
      }
      return next;
    });
  };

  // Handle section edit
  const handleEdit = async (section: VideoSection) => {
    if (editingSection === section.id) {
      // Save changes
      if (onSectionUpdate) {
        await onSectionUpdate(section);
      }
      setEditingSection(null);
    } else {
      setEditingSection(section.id);
    }
  };

  // Handle section delete
  const handleDelete = async (sectionId: string) => {
    if (onSectionDelete && window.confirm('Are you sure you want to delete this section?')) {
      await onSectionDelete(sectionId);
    }
  };

  // Handle new section creation
  const handleCreate = async () => {
    if (onSectionCreate && newSection.title && newSection.start_time !== undefined && newSection.end_time !== undefined) {
      await onSectionCreate({
        title: newSection.title,
        start_time: newSection.start_time,
        end_time: newSection.end_time,
        type: 'manual',
        description: newSection.description
      });
      setNewSection({
        title: '',
        start_time: 0,
        end_time: 0,
        type: 'manual'
      });
    }
  };

  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm">
      {/* Header */}
      <div className="p-4 border-b dark:border-gray-700">
        <h3 className="text-lg font-semibold text-gray-900 dark:text-white flex items-center">
          <Clock className="h-5 w-5 mr-2 text-primary dark:text-primary-light" />
          Video Sections
        </h3>
      </div>

      {/* Sections List */}
      <div className="divide-y dark:divide-gray-700">
        {sections.map(section => (
          <div
            key={section.id}
            className={`p-4 transition-colors ${
              currentSection?.id === section.id
                ? 'bg-blue-50 dark:bg-blue-900/20'
                : 'hover:bg-gray-50 dark:hover:bg-gray-700/50'
            }`}
          >
            {/* Section Header */}
            <div className="flex items-center justify-between">
              <div className="flex-1 min-w-0">
                <div className="flex items-center space-x-2">
                  <button
                    onClick={() => toggleSection(section.id)}
                    className="text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200"
                  >
                    {expandedSections.has(section.id) ? (
                      <ChevronUp className="h-4 w-4" />
                    ) : (
                      <ChevronDown className="h-4 w-4" />
                    )}
                  </button>
                  <span className="text-sm font-medium text-gray-900 dark:text-white">
                    {section.title}
                  </span>
                  <span className="text-xs text-gray-500 dark:text-gray-400">
                    {formatTime(section.start_time)} - {formatTime(section.end_time)}
                  </span>
                </div>
              </div>
              <div className="flex items-center space-x-2">
                <button
                  onClick={() => onSeek(section.start_time)}
                  className="text-primary dark:text-primary-light hover:text-primary-dark dark:hover:text-primary-light/80 text-sm"
                >
                  Jump to
                </button>
                {isEditing && (
                  <>
                    <button
                      onClick={() => handleEdit(section)}
                      className="text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200"
                    >
                      <Edit2 className="h-4 w-4" />
                    </button>
                    <button
                      onClick={() => handleDelete(section.id)}
                      className="text-red-500 hover:text-red-700 dark:text-red-400 dark:hover:text-red-300"
                    >
                      <Trash2 className="h-4 w-4" />
                    </button>
                  </>
                )}
              </div>
            </div>

            {/* Section Details */}
            {expandedSections.has(section.id) && (
              <div className="mt-2 pl-6">
                {section.description && (
                  <p className="text-sm text-gray-600 dark:text-gray-300 mb-2">
                    {section.description}
                  </p>
                )}
                <div className="text-xs text-gray-500 dark:text-gray-400">
                  Type: {section.type} • Confidence: {(section.confidence * 100).toFixed(0)}%
                </div>
              </div>
            )}
          </div>
        ))}
      </div>

      {/* New Section Form */}
      {isEditing && (
        <div className="p-4 border-t dark:border-gray-700">
          <h4 className="text-sm font-medium text-gray-900 dark:text-white mb-3">
            Add New Section
          </h4>
          <div className="space-y-3">
            <input
              type="text"
              value={newSection.title}
              onChange={e => setNewSection(prev => ({ ...prev, title: e.target.value }))}
              placeholder="Section title"
              className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md text-sm bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
            />
            <div className="grid grid-cols-2 gap-2">
              <input
                type="number"
                value={newSection.start_time}
                onChange={e => setNewSection(prev => ({ ...prev, start_time: parseFloat(e.target.value) }))}
                placeholder="Start time (seconds)"
                className="px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md text-sm bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
              />
              <input
                type="number"
                value={newSection.end_time}
                onChange={e => setNewSection(prev => ({ ...prev, end_time: parseFloat(e.target.value) }))}
                placeholder="End time (seconds)"
                className="px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md text-sm bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
              />
            </div>
            <textarea
              value={newSection.description}
              onChange={e => setNewSection(prev => ({ ...prev, description: e.target.value }))}
              placeholder="Section description (optional)"
              className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md text-sm bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
              rows={2}
            />
            <button
              onClick={handleCreate}
              disabled={!newSection.title || newSection.start_time === undefined || newSection.end_time === undefined}
              className="w-full flex items-center justify-center px-4 py-2 bg-primary dark:bg-primary-light text-white rounded-md hover:bg-primary-dark dark:hover:bg-primary-light/90 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              <Plus className="h-4 w-4 mr-2" />
              Add Section
            </button>
          </div>
        </div>
      )}
    </div>
  );
}; 