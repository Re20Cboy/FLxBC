// SPDX-License-Identifier: MIT
pragma solidity ^0.8.28;

contract FederationRegistry {
    address public owner;

    struct FederationNode {
        string nodeId;
        string displayName;
        uint256 reputationBps;
        bool active;
        uint256 registeredAt;
    }

    mapping(bytes32 => FederationNode) public nodes;
    bytes32[] public nodeKeys;

    event NodeRegistered(string indexed nodeId, string displayName, uint256 reputationBps);
    event NodeStatusChanged(string indexed nodeId, bool active, uint256 reputationBps);

    modifier onlyOwner() {
        require(msg.sender == owner, "not owner");
        _;
    }

    constructor() {
        owner = msg.sender;
    }

    function registerNode(
        string calldata nodeId,
        string calldata displayName,
        uint256 reputationBps
    ) external onlyOwner {
        bytes32 key = keccak256(bytes(nodeId));
        require(bytes(nodes[key].nodeId).length == 0, "node exists");
        nodes[key] = FederationNode({
            nodeId: nodeId,
            displayName: displayName,
            reputationBps: reputationBps,
            active: true,
            registeredAt: block.timestamp
        });
        nodeKeys.push(key);
        emit NodeRegistered(nodeId, displayName, reputationBps);
    }

    function setNodeStatus(
        string calldata nodeId,
        bool active,
        uint256 reputationBps
    ) external onlyOwner {
        bytes32 key = keccak256(bytes(nodeId));
        require(bytes(nodes[key].nodeId).length != 0, "unknown node");
        nodes[key].active = active;
        nodes[key].reputationBps = reputationBps;
        emit NodeStatusChanged(nodeId, active, reputationBps);
    }

    function nodeCount() external view returns (uint256) {
        return nodeKeys.length;
    }
}
