# Chess Move Translation

Move translation uses this priority order: castling, en passant, promotion, capture, normal. This avoids misclassifying en passant as a regular capture and promotion-captures as normal captures.

Normal moves produce one physical action. Captures move the captured piece to its color graveyard before moving the attacker. Castling moves the king first and rook second. En passant removes the captured pawn from its actual square. Promotion stores the pawn in pawn storage and activates a reserve piece; promotion-capture first clears the captured piece.

Logical chess state is committed only after all physical actions succeed.
