import React, { useState, useEffect } from 'react';
import { useGameStore } from './stores/gameStore';
import { useWebSocket } from './hooks/useWebSocket';
import { PokerTable } from './components/PokerTable/PokerTable';
import { ActionPanel } from './components/Controls/ActionPanel';
import { HandHistory } from './components/Sidebar/HandHistory';
import { OpponentStats } from './components/Sidebar/OpponentStats';
import { NewGameModal } from './components/Modals/NewGameModal';
import { HandResultModal } from './components/Modals/HandResultModal';
import { ToastContainer, toast } from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';

function App() {
  const { gameState, sessionId, isConnected, error, startNextHand } = useGameStore();
  const [showNewGameModal, setShowNewGameModal] = useState(true);

  // Connect WebSocket when session is created
  useWebSocket(sessionId);

  // Show error toasts
  useEffect(() => {
    if (error) {
      toast.error(error);
    }
  }, [error]);

  const handleNewGame = () => {
    setShowNewGameModal(true);
  };

  const handleNextHand = async () => {
    await startNextHand();
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 to-gray-800">
      <ToastContainer position="top-right" theme="dark" />

      {/* Header */}
      <header className="bg-gray-900 border-b border-gray-700 p-4">
        <div className="container mx-auto flex justify-between items-center">
          <h1 className="text-white text-3xl font-bold">LagBot Poker</h1>
          <div className="flex gap-4 items-center">
            {isConnected && (
              <span className="text-green-400 text-sm">Connected</span>
            )}
            <button
              onClick={handleNewGame}
              className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg"
            >
              New Game
            </button>
          </div>
        </div>
      </header>

      {/* Main content */}
      <main className="container mx-auto p-8">
        {!gameState ? (
          <div className="flex items-center justify-center h-[600px]">
            <div className="text-white text-xl">
              Create a new game to start playing
            </div>
          </div>
        ) : (
          <div className="flex gap-6">
            {/* Left sidebar */}
            <div className="flex flex-col gap-4 w-64">
              <HandHistory gameState={gameState} />
              <OpponentStats players={gameState.players} />
            </div>

            {/* Center - Poker table */}
            <div className="flex-1 flex flex-col gap-4 items-center">
              <PokerTable gameState={gameState} />
              <ActionPanel gameState={gameState} className="w-full max-w-2xl" />
            </div>

            {/* Right sidebar - could add more stats or info */}
            <div className="w-64">
              <div className="bg-gray-800 rounded-lg p-4">
                <h3 className="text-white font-bold mb-2">Game Info</h3>
                <div className="text-gray-300 text-sm space-y-1">
                  <div>Hand: #{gameState.hand_number}</div>
                  <div>Small Blind: ${gameState.small_blind}</div>
                  <div>Big Blind: ${gameState.big_blind}</div>
                  <div>Players: {gameState.players.length}</div>
                </div>
              </div>
            </div>
          </div>
        )}
      </main>

      {/* Modals */}
      <NewGameModal
        isOpen={showNewGameModal}
        onClose={() => setShowNewGameModal(false)}
      />

      {gameState && gameState.hand_complete && (
        <HandResultModal gameState={gameState} onNextHand={handleNextHand} />
      )}
    </div>
  );
}

export default App;
