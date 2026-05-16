// SPDX-License-Identifier: MIT
pragma solidity ^0.8.28;

contract MisbehaviorRegistry {
    address public owner;

    struct MisbehaviorEvent {
        string runId;
        uint256 roundId;
        string nodeId;
        string kind;
        uint256 penaltyBps;
        string evidenceHash;
        uint256 reportedAt;
    }

    MisbehaviorEvent[] public events_;

    event MisbehaviorReported(
        string indexed runId,
        uint256 indexed roundId,
        string indexed nodeId,
        string kind,
        uint256 penaltyBps,
        string evidenceHash
    );

    modifier onlyOwner() {
        require(msg.sender == owner, "not owner");
        _;
    }

    constructor() {
        owner = msg.sender;
    }

    function report(
        string calldata runId,
        uint256 roundId,
        string calldata nodeId,
        string calldata kind,
        uint256 penaltyBps,
        string calldata evidenceHash
    ) external onlyOwner {
        events_.push(MisbehaviorEvent({
            runId: runId,
            roundId: roundId,
            nodeId: nodeId,
            kind: kind,
            penaltyBps: penaltyBps,
            evidenceHash: evidenceHash,
            reportedAt: block.timestamp
        }));
        emit MisbehaviorReported(runId, roundId, nodeId, kind, penaltyBps, evidenceHash);
    }

    function eventCount() external view returns (uint256) {
        return events_.length;
    }
}
