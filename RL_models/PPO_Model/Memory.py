import numpy as np
from checkers_game.constants import NUM_ACTIONS


class Memory:
    """Experience replay buffer backed by pre-allocated numpy arrays.

    Uses a doubling strategy on overflow so the amortised append cost is O(1).
    Properties return zero-copy views into the backing arrays, making the
    np.asarray() conversion in Agent.update() a no-op rather than an O(N) copy.

    The `values` field stores the old value estimate produced by the inference
    server at collection time, used for PPO value function clipping in update().
    """

    # Per-game Memory objects only need ~75 entries (half the moves in one game).
    # The epoch-level combined Memory grows to 300K+ via repeated doubling from here.
    # Keeping this small avoids exhausting Windows kernel pool memory when 24 workers
    # simultaneously allocate and pickle two Memory objects each through IPC queues.
    _INITIAL_CAP = 128

    def __init__(self):
        cap = self._INITIAL_CAP
        self._states       = np.empty((cap, 4, 8, 8), dtype=np.float32)
        self._actions      = np.empty(cap,             dtype=np.int64)
        self._rewards      = np.empty(cap,             dtype=np.float32)
        self._log_probs    = np.empty(cap,             dtype=np.float32)
        self._done         = np.empty(cap,             dtype=np.bool_)
        self._action_masks = np.empty((cap, NUM_ACTIONS), dtype=np.float32)
        self._values       = np.empty(cap,             dtype=np.float32)
        self._size = 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _grow(self):
        """Double all backing arrays, preserving existing data."""
        old     = self._size
        new_cap = len(self._states) * 2

        def _ext1d(a):
            n = np.empty(new_cap, dtype=a.dtype)
            n[:old] = a[:old]
            return n

        def _ext2d(a, d):
            n = np.empty((new_cap, d), dtype=a.dtype)
            n[:old] = a[:old]
            return n

        def _extnd(a, shape):
            n = np.empty((new_cap,) + shape, dtype=a.dtype)
            n[:old] = a[:old]
            return n

        self._states       = _extnd(self._states,       (4, 8, 8))
        self._actions      = _ext1d(self._actions)
        self._rewards      = _ext1d(self._rewards)
        self._log_probs    = _ext1d(self._log_probs)
        self._done         = _ext1d(self._done)
        self._action_masks = _ext2d(self._action_masks, NUM_ACTIONS)
        self._values       = _ext1d(self._values)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(self, state, action_index, reward, log_prob, done,
            action_mask=None, value=0.0):
        if self._size >= len(self._states):
            self._grow()
        i = self._size
        self._states[i]       = state
        self._actions[i]      = action_index
        self._rewards[i]      = reward
        self._log_probs[i]    = log_prob
        self._done[i]         = done
        self._action_masks[i] = action_mask if action_mask is not None else 0.0
        self._values[i]       = value
        self._size += 1

    def extend(self, other):
        """Append all entries from another Memory. Grows backing arrays as needed."""
        n = other._size
        if n == 0:
            return
        while self._size + n > len(self._states):
            self._grow()
        s = self._size
        self._states      [s:s+n] = other._states      [:n]
        self._actions     [s:s+n] = other._actions     [:n]
        self._rewards     [s:s+n] = other._rewards     [:n]
        self._log_probs   [s:s+n] = other._log_probs   [:n]
        self._done        [s:s+n] = other._done        [:n]
        self._action_masks[s:s+n] = other._action_masks[:n]
        self._values      [s:s+n] = other._values      [:n]
        self._size += n

    def clear(self):
        """Reset the size pointer without reallocating."""
        self._size = 0

    def update_last_done(self):
        if self._size > 0:
            self._done[self._size - 1] = True

    # ------------------------------------------------------------------
    # Zero-copy property views
    # ------------------------------------------------------------------

    @property
    def states(self):       return self._states      [:self._size]
    @property
    def actions(self):      return self._actions     [:self._size]
    @property
    def rewards(self):      return self._rewards     [:self._size]
    @property
    def log_probs(self):    return self._log_probs   [:self._size]
    @property
    def done(self):         return self._done        [:self._size]
    @property
    def action_masks(self): return self._action_masks[:self._size]
    @property
    def values(self):       return self._values      [:self._size]

    def __len__(self):
        return self._size
