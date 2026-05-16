// SPDX-License-Identifier: MIT
pragma solidity ^0.8.28;

contract TrainingLedger {
    address public owner;

    struct RoundRecord {
        string runId;
        uint256 roundId;
        bytes32 modelHash;
        bytes32 metricsHash;
        bytes32 participantsHash;
        bytes32 strategyHash;
        string artifactURI;
        uint256 committedAt;
    }

    mapping(bytes32 => RoundRecord) public rounds;

    event RoundCommitted(
        string indexed runId,
        uint256 indexed roundId,
        bytes32 modelHash,
        bytes32 metricsHash,
        bytes32 participantsHash,
        bytes32 strategyHash,
        string artifactURI
    );

    modifier onlyOwner() {
        require(msg.sender == owner, "not owner");
        _;
    }

    constructor() {
        owner = msg.sender;
    }

    function commitRound(
        string calldata runId,
        uint256 roundId,
        bytes32 modelHash,
        bytes32 metricsHash,
        bytes32 participantsHash,
        bytes32 strategyHash,
        string calldata artifactURI
    ) external onlyOwner {
        bytes32 key = keccak256(abi.encodePacked(runId, roundId));
        require(rounds[key].committedAt == 0, "round exists");
        rounds[key] = RoundRecord({
            runId: runId,
            roundId: roundId,
            modelHash: modelHash,
            metricsHash: metricsHash,
            participantsHash: participantsHash,
            strategyHash: strategyHash,
            artifactURI: artifactURI,
            committedAt: block.timestamp
        });
        emit RoundCommitted(
            runId,
            roundId,
            modelHash,
            metricsHash,
            participantsHash,
            strategyHash,
            artifactURI
        );
    }
}
