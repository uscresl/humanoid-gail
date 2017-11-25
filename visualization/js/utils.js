function vectorize(vector)
{
	return new THREE.Vector3(vector[0], vector[1], vector[2]);
}

function loadSubject(scene, subject)
{
	var bones = {};
	function dfs(position, node)
	{
		node.position = position;
		node.C = math.eye(3);
		node.Cinv = math.eye(3);
		if ("direction" in node)
		{
			var direction = vectorize(node.direction);
			direction.multiplyScalar(node.length);
			var endPosition = new THREE.Vector3().addVectors(position, direction);
			if (node.name != "root")
			{
				var geometry = new THREE.SphereGeometry(node.name == "head" ? 2.5 : 1, 32, 32);
				var material = new THREE.MeshPhongMaterial( {
					color: 0x156289,
					emissive: 0x072534,
					side: THREE.DoubleSide,
					flatShading: true
				} );
				var sphere = new THREE.Mesh( geometry, material );
				sphere.name = node.name;
				sphere.castShadow = true;
				sphere.receiveShadow = true;
				scene.add(sphere);

				var rot = rotation_matrix_axis(node.axis);
				node.C = rot.C;
				node.Cinv = rot.Cinv;
			}
			for (var name in node.children)
			{
				dfs(endPosition, node.children[name]);
			}
		}
		else
		{
			for (var name in node.children)
			{
				dfs(position, node.children[name]);
			}
		}
		bones[node.name] = node;
	}

	dfs(new THREE.Vector3(), subject.root);

	return {
		subject: subject,
		bones: bones
	};
}

function loadAnimation()
{
	return new Promise(function(resolve, reject)
	{
		$.get("mocap/animation.json", function(animation)
		{
			resolve(animation.frames);
		});
	});
}

// math util functions
function deg2rad(deg)
{
	return deg * Math.PI / 180.;
}

// This code is inspired by https://github.com/VanushVaswani/amcparser
function rotation_matrix_axis(C_values)
{
	// Change coordinate system through matrix C
	var rx = deg2rad(C_values[0]);
	var ry = deg2rad(C_values[1]);
	var rz = deg2rad(C_values[2]);

	var Cx = math.matrix([[1, 0, 0],
					[0, math.cos(rx), math.sin(rx)],
					[0, -math.sin(rx), math.cos(rx)]]);

	var Cy = math.matrix([[math.cos(ry), 0, -math.sin(ry)],
					[0, 1, 0],
					[math.sin(ry), 0, math.cos(ry)]]);

	var Cz = math.matrix([[math.cos(rz), math.sin(rz), 0],
					[-math.sin(rz), math.cos(rz), 0],
					[0, 0, 1]]);

	var C = math.multiply(math.multiply(Cx, Cy), Cz);
	var Cinv = math.inv(C);
	return {
		C: C,
		Cinv: Cinv
	};
}

function rotation_matrix(bone, tx, ty, tz)
{
	// Construct rotation matrix M
	var tx = deg2rad(tx);
	var ty = deg2rad(ty);
	var tz = deg2rad(tz);

	var Mx = math.matrix([[1, 0, 0],
					[0, math.cos(tx), math.sin(tx)],
					[0, -math.sin(tx), math.cos(tx)]]);

	var My = math.matrix([[math.cos(ty), 0, -math.sin(ty)],
					[0, 1, 0],
					[math.sin(ty), 0, math.cos(ty)]]);

	var Mz = math.matrix([[math.cos(tz), math.sin(tz), 0],
					[-math.sin(tz), math.cos(tz), 0],
					[0, 0, 1]]);
	var M = math.multiply(math.multiply(Mx, My), Mz);
	var L = math.multiply(math.multiply(bone.Cinv, M), bone.C);
	return L;
}
