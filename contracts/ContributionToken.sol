// SPDX-License-Identifier: MIT
pragma solidity ^0.8.28;

contract ContributionToken {
    address public owner;
    string public name = "FLxBC Contribution Point";
    string public symbol = "FLXBCP";
    uint8 public decimals = 0;
    uint256 public totalSupply;

    mapping(address => uint256) public balanceOf;

    event ContributionSettled(address indexed node, uint256 amount, string reason);

    modifier onlyOwner() {
        require(msg.sender == owner, "not owner");
        _;
    }

    constructor() {
        owner = msg.sender;
    }

    function mint(address node, uint256 amount, string calldata reason) external onlyOwner {
        balanceOf[node] += amount;
        totalSupply += amount;
        emit ContributionSettled(node, amount, reason);
    }

    function transfer(address, uint256) external pure returns (bool) {
        revert("non-transferable");
    }

    function approve(address, uint256) external pure returns (bool) {
        revert("non-transferable");
    }
}
