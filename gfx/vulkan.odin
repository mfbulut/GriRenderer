package gfx

import "core:mem"
import "core:dynlib"
import "core:math"
import "core:math/linalg"
import "core:image"
import _ "core:image/png"
import _ "core:image/jpeg"
import vk "vendor:vulkan"

MAX_VERTICES  :: 8_000_000
MAX_INDICES   :: 24_000_000
MAX_INSTANCES :: 96_000
MAX_TEXTURES  :: 1024
STAGING_SIZE  :: 256 * mem.Megabyte
BLAS_SIZE     :: 64 * mem.Megabyte

Push_Constants :: struct #packed {
	view_proj:         Mat4,
	instances_address: vk.DeviceAddress,
}

Vertex :: struct #packed {
	pos:    Vec3,
	normal: Vec3,
	uv:     Vec2,
	color:  Color,
}

Gpu_Instance :: struct #packed {
	model_mat: Mat4,
	color:     Color,
	texture:   u32,
	type:      u32,
}

GPU_Mesh :: struct {
	index_count:   u32,
	index_offset:  u32,
	vertex_offset: i32,
	blas_address: vk.DeviceAddress,
}

GPU_Buffer :: struct {
	handle:  vk.Buffer,
	memory:  vk.DeviceMemory,
	mapped:  rawptr,
	size:    vk.DeviceSize,
	address: vk.DeviceAddress,
}

GPU_Texture :: struct {
	index:      u32,
	image:      vk.Image,
	memory:     vk.DeviceMemory,
	view:       vk.ImageView,
	format:     vk.Format,
	width:      u32,
	height:     u32,
	mip_levels: u32,
}

vks: struct {
	gpu:          vk.PhysicalDevice,
	device:       vk.Device,
	queue:        vk.Queue,
	queue_family: u32,

	surface:          vk.SurfaceKHR,
	swapchain:        vk.SwapchainKHR,
	swapchain_images: []vk.Image,
	swapchain_views:  []vk.ImageView,
	render_done:      []vk.Semaphore,
	image_acquired:   vk.Semaphore,
	in_flight:        vk.Fence,
	msaa_image:       GPU_Texture,
	depth_image:      GPU_Texture,

	cmd_pool:              vk.CommandPool,
	cmd_buf:               vk.CommandBuffer,
	upload_cmd_pool:       vk.CommandPool,
	upload_cmd:            vk.CommandBuffer,
	upload_staging_offset: vk.DeviceSize,
	descriptor_set:        vk.DescriptorSet,
	pipeline_layout:       vk.PipelineLayout,
	pipeline:              vk.Pipeline,
	sampler:               vk.Sampler,

	vertex_buffer:       GPU_Buffer,
	index_buffer:        GPU_Buffer,
	instance_buffer:     GPU_Buffer,
	indirect_buffer:     GPU_Buffer,
	staging_buffer:      GPU_Buffer,
	blas_scratch_buffer: GPU_Buffer,

	vertex_count_total: u32,
	index_count_total:  u32,
	texture_count:      u32,

	tlas:                  vk.AccelerationStructureKHR,
	tlas_buffer:           GPU_Buffer,
	tlas_scratch_buffer:   GPU_Buffer,
	tlas_instances_buffer: GPU_Buffer,

	blas_addresses: [dynamic; MAX_INSTANCES]vk.DeviceAddress,
	gpu_instances:  [dynamic; MAX_INSTANCES]Gpu_Instance,
	indirect_cmds:  [dynamic; MAX_INSTANCES]vk.DrawIndexedIndirectCommand,

	sky_mesh:    GPU_Mesh,
	sky_texture: GPU_Texture,
}

camera: struct {
	pos:    Vec3,
	target: Vec3,
	up:     Vec3,
	fov:    f32,
	near:   f32,
	far:    f32,
}

find_memory_type :: proc(type_filter: u32, properties: vk.MemoryPropertyFlags) -> u32 {
	mem_props: vk.PhysicalDeviceMemoryProperties
	vk.GetPhysicalDeviceMemoryProperties(vks.gpu, &mem_props)
	for i in 0..<mem_props.memoryTypeCount {
		if (type_filter & (1 << i)) != 0 && (mem_props.memoryTypes[i].propertyFlags & properties) == properties {
			return i
		}
	}
	return 0
}

buffer_create :: proc(size: vk.DeviceSize, usage: vk.BufferUsageFlags, host_visible := false) -> (buf: GPU_Buffer) {
	buf_info := vk.BufferCreateInfo{
		sType       = .BUFFER_CREATE_INFO,
		size        = size,
		usage       = usage + {.SHADER_DEVICE_ADDRESS},
		sharingMode = .EXCLUSIVE,
	}
	vk.CreateBuffer(vks.device, &buf_info, nil, &buf.handle)

	mem_reqs: vk.MemoryRequirements
	vk.GetBufferMemoryRequirements(vks.device, buf.handle, &mem_reqs)

	props: vk.MemoryPropertyFlags = {.HOST_VISIBLE, .HOST_COHERENT} if host_visible else {.DEVICE_LOCAL}
	mem_type := find_memory_type(mem_reqs.memoryTypeBits, props)

	alloc_flags := vk.MemoryAllocateFlagsInfo{
		sType = .MEMORY_ALLOCATE_FLAGS_INFO,
		flags = {.DEVICE_ADDRESS},
	}
	alloc_info := vk.MemoryAllocateInfo{
		sType           = .MEMORY_ALLOCATE_INFO,
		pNext           = &alloc_flags,
		allocationSize  = mem_reqs.size,
		memoryTypeIndex = mem_type,
	}
	vk.AllocateMemory(vks.device, &alloc_info, nil, &buf.memory)
	vk.BindBufferMemory(vks.device, buf.handle, buf.memory, 0)
	buf.size = size

	if host_visible {
		vk.MapMemory(vks.device, buf.memory, 0, size, {}, &buf.mapped)
	}

	addr_info := vk.BufferDeviceAddressInfo{
		sType  = .BUFFER_DEVICE_ADDRESS_INFO,
		buffer = buf.handle,
	}
	buf.address = vk.GetBufferDeviceAddress(vks.device, &addr_info)

	return buf
}

buffer_destroy :: proc(buf: ^GPU_Buffer) {
	if buf.mapped != nil {
		vk.UnmapMemory(vks.device, buf.memory)
		buf.mapped = nil
	}
	if buf.handle != 0 {
		vk.DestroyBuffer(vks.device, buf.handle, nil)
		buf.handle = 0
	}
	if buf.memory != 0 {
		vk.FreeMemory(vks.device, buf.memory, nil)
		buf.memory = 0
	}
}

upload_begin :: proc() {
	cmd_alloc := vk.CommandBufferAllocateInfo{
		sType              = .COMMAND_BUFFER_ALLOCATE_INFO,
		commandPool        = vks.upload_cmd_pool,
		level              = .PRIMARY,
		commandBufferCount = 1,
	}
	vk.AllocateCommandBuffers(vks.device, &cmd_alloc, &vks.upload_cmd)

	begin_info := vk.CommandBufferBeginInfo{
		sType = .COMMAND_BUFFER_BEGIN_INFO,
		flags = {.ONE_TIME_SUBMIT},
	}
	vk.BeginCommandBuffer(vks.upload_cmd, &begin_info)
	vks.upload_staging_offset = 0
}

staging_alloc :: proc(size: vk.DeviceSize, alignment: vk.DeviceSize = 16) -> (rawptr, vk.DeviceSize) {
	offset := (vks.upload_staging_offset + (alignment - 1)) &~ (alignment - 1)
	if offset + size > STAGING_SIZE {
		upload_end()
		upload_begin()
		offset = 0
	}
	assert(offset + size <= STAGING_SIZE, "Staging buffer overflow: asset size exceeds STAGING_SIZE")
	vks.upload_staging_offset = offset + size
	ptr := rawptr(uintptr(vks.staging_buffer.mapped) + uintptr(offset))
	return ptr, offset
}

upload_end :: proc() {
	vk.EndCommandBuffer(vks.upload_cmd)

	cmd_submit := vk.CommandBufferSubmitInfo{
		sType         = .COMMAND_BUFFER_SUBMIT_INFO,
		commandBuffer = vks.upload_cmd,
	}
	submit_info := vk.SubmitInfo2{
		sType                  = .SUBMIT_INFO_2,
		commandBufferInfoCount = 1,
		pCommandBufferInfos    = &cmd_submit,
	}
	vk.QueueSubmit2(vks.queue, 1, &submit_info, 0)
	vk.QueueWaitIdle(vks.queue)

	vk.FreeCommandBuffers(vks.device, vks.upload_cmd_pool, 1, &vks.upload_cmd)
	vks.upload_staging_offset = 0
	vks.upload_cmd = nil
}

image_barrier :: proc(
	cmd: vk.CommandBuffer, img: vk.Image,
	old_layout, new_layout: vk.ImageLayout,
	src_stage, dst_stage: vk.PipelineStageFlags2,
	src_access, dst_access: vk.AccessFlags2,
	aspect_mask: vk.ImageAspectFlags = {.COLOR},
	mip_levels: u32 = 1, base_mip: u32 = 0,
) {
	barrier := vk.ImageMemoryBarrier2{
		sType               = .IMAGE_MEMORY_BARRIER_2,
		srcStageMask        = src_stage,
		srcAccessMask       = src_access,
		dstStageMask        = dst_stage,
		dstAccessMask       = dst_access,
		oldLayout           = old_layout,
		newLayout           = new_layout,
		srcQueueFamilyIndex = vk.QUEUE_FAMILY_IGNORED,
		dstQueueFamilyIndex = vk.QUEUE_FAMILY_IGNORED,
		image               = img,
		subresourceRange    = {
			aspectMask     = aspect_mask,
			baseMipLevel   = base_mip,
			levelCount     = mip_levels,
			baseArrayLayer = 0,
			layerCount     = 1,
		},
	}
	dep_info := vk.DependencyInfo{
		sType                   = .DEPENDENCY_INFO,
		imageMemoryBarrierCount = 1,
		pImageMemoryBarriers    = &barrier,
	}
	vk.CmdPipelineBarrier2(cmd, &dep_info)
}

texture_create :: proc(
	width, height: u32,
	format: vk.Format,
	usage: vk.ImageUsageFlags,
	samples: vk.SampleCountFlags = {._1},
	mip_levels: u32 = 1,
	aspect: vk.ImageAspectFlags = {.COLOR},
) -> (tex: GPU_Texture) {
	img_info := vk.ImageCreateInfo{
		sType         = .IMAGE_CREATE_INFO,
		imageType     = .D2,
		extent        = { width = width, height = height, depth = 1 },
		mipLevels     = mip_levels,
		arrayLayers   = 1,
		format        = format,
		tiling        = .OPTIMAL,
		initialLayout = .UNDEFINED,
		usage         = usage,
		samples       = samples,
		sharingMode   = .EXCLUSIVE,
	}
	vk.CreateImage(vks.device, &img_info, nil, &tex.image)

	mem_reqs: vk.MemoryRequirements
	vk.GetImageMemoryRequirements(vks.device, tex.image, &mem_reqs)
	mem_type := find_memory_type(mem_reqs.memoryTypeBits, {.DEVICE_LOCAL})

	alloc_info := vk.MemoryAllocateInfo{
		sType           = .MEMORY_ALLOCATE_INFO,
		allocationSize  = mem_reqs.size,
		memoryTypeIndex = mem_type,
	}
	vk.AllocateMemory(vks.device, &alloc_info, nil, &tex.memory)
	vk.BindImageMemory(vks.device, tex.image, tex.memory, 0)

	view_info := vk.ImageViewCreateInfo{
		sType    = .IMAGE_VIEW_CREATE_INFO,
		image    = tex.image,
		viewType = .D2,
		format   = format,
		subresourceRange = {
			aspectMask     = aspect,
			levelCount     = mip_levels,
			layerCount     = 1,
		},
	}
	vk.CreateImageView(vks.device, &view_info, nil, &tex.view)

	tex.format     = format
	tex.width      = width
	tex.height     = height
	tex.mip_levels = mip_levels
	return tex
}

texture_destroy :: proc(tex: ^GPU_Texture) {
	if tex.view != 0 {
		vk.DestroyImageView(vks.device, tex.view, nil)
		tex.view = 0
	}
	if tex.image != 0 {
		vk.DestroyImage(vks.device, tex.image, nil)
		tex.image = 0
	}
	if tex.memory != 0 {
		vk.FreeMemory(vks.device, tex.memory, nil)
		tex.memory = 0
	}
}

mipmaps_generate :: proc(cmd: vk.CommandBuffer, img: vk.Image, tex_width, tex_height: i32, mip_levels: u32) {
	mip_w := tex_width
	mip_h := tex_height

	for i in 1..<mip_levels {
		image_barrier(
			cmd, img,
			.TRANSFER_DST_OPTIMAL, .TRANSFER_SRC_OPTIMAL,
			{.TRANSFER}, {.TRANSFER},
			{.TRANSFER_WRITE}, {.TRANSFER_READ},
			{.COLOR}, 1, i - 1,
		)

		next_w := max(mip_w / 2, 1)
		next_h := max(mip_h / 2, 1)

		blit := vk.ImageBlit{
			srcSubresource = { aspectMask = {.COLOR}, mipLevel = i - 1, layerCount = 1 },
			srcOffsets      = { 1 = {mip_w, mip_h, 1} },
			dstSubresource = { aspectMask = {.COLOR}, mipLevel = i, layerCount = 1 },
			dstOffsets      = { 1 = {next_w, next_h, 1} },
		}
		vk.CmdBlitImage(cmd, img, .TRANSFER_SRC_OPTIMAL, img, .TRANSFER_DST_OPTIMAL, 1, &blit, .LINEAR)

		image_barrier(
			cmd, img,
			.TRANSFER_SRC_OPTIMAL, .SHADER_READ_ONLY_OPTIMAL,
			{.TRANSFER}, {.FRAGMENT_SHADER},
			{.TRANSFER_READ}, {.SHADER_READ, .SHADER_SAMPLED_READ},
			{.COLOR}, 1, i - 1,
		)

		mip_w = next_w
		mip_h = next_h
	}

	image_barrier(
		cmd, img,
		.TRANSFER_DST_OPTIMAL, .SHADER_READ_ONLY_OPTIMAL,
		{.TRANSFER}, {.FRAGMENT_SHADER},
		{.TRANSFER_WRITE}, {.SHADER_READ, .SHADER_SAMPLED_READ},
		{.COLOR}, 1, mip_levels - 1,
	)
}

texture_upload :: proc(tex: ^GPU_Texture, pixels: []u8, generate_mips := true) {
	img_size := vk.DeviceSize(tex.width * tex.height * 4)
	staging_ptr, staging_off := staging_alloc(img_size, 16)
	mem.copy(staging_ptr, raw_data(pixels), int(img_size))

	image_barrier(
		vks.upload_cmd, tex.image,
		.UNDEFINED, .TRANSFER_DST_OPTIMAL,
		{.ALL_COMMANDS}, {.TRANSFER},
		{}, {.TRANSFER_WRITE},
		{.COLOR}, tex.mip_levels, 0,
	)

	region := vk.BufferImageCopy{
		bufferOffset      = staging_off,
		imageSubresource  = { aspectMask = {.COLOR}, layerCount = 1 },
		imageExtent       = { tex.width, tex.height, 1 },
	}
	vk.CmdCopyBufferToImage(vks.upload_cmd, vks.staging_buffer.handle, tex.image, .TRANSFER_DST_OPTIMAL, 1, &region)

	if !generate_mips || tex.mip_levels <= 1 {
		image_barrier(
			vks.upload_cmd, tex.image,
			.TRANSFER_DST_OPTIMAL, .SHADER_READ_ONLY_OPTIMAL,
			{.TRANSFER}, {.FRAGMENT_SHADER},
			{.TRANSFER_WRITE}, {.SHADER_READ, .SHADER_SAMPLED_READ},
			{.COLOR}, tex.mip_levels, 0,
		)
	} else {
		mipmaps_generate(vks.upload_cmd, tex.image, i32(tex.width), i32(tex.height), tex.mip_levels)
	}

	assert(vks.texture_count < MAX_TEXTURES, "Texture limit exceeded: MAX_TEXTURES reached")
	tex_id := vks.texture_count
	vks.texture_count += 1
	tex.index = tex_id

	image_desc_info := vk.DescriptorImageInfo{
		sampler     = vks.sampler,
		imageView   = tex.view,
		imageLayout = .SHADER_READ_ONLY_OPTIMAL,
	}
	write_desc := vk.WriteDescriptorSet{
		sType           = .WRITE_DESCRIPTOR_SET,
		dstSet          = vks.descriptor_set,
		dstBinding      = 0,
		dstArrayElement = tex_id,
		descriptorType  = .COMBINED_IMAGE_SAMPLER,
		descriptorCount = 1,
		pImageInfo      = &image_desc_info,
	}
	vk.UpdateDescriptorSets(vks.device, 1, &write_desc, 0, nil)
}

texture_create_from_pixels :: proc(width, height: u32, pixels: []u8, generate_mips := true) -> GPU_Texture {
	mips: u32 = 1
	if generate_mips {
		mips = u32(math.floor(math.log2(f32(max(width, height))))) + 1
	}
	tex := texture_create(width, height, .R8G8B8A8_SRGB, {.TRANSFER_SRC, .TRANSFER_DST, .SAMPLED}, {._1}, mips, {.COLOR})
	texture_upload(&tex, pixels, generate_mips)
	return tex
}

texture_create_from_memory :: proc(data: []byte, generate_mips := true) -> (tex: GPU_Texture, ok: bool) {
	img, err := image.load_from_bytes(data, {.alpha_add_if_missing}, context.temp_allocator)
	if err != nil || img == nil do return {}, false
	defer image.destroy(img, context.temp_allocator)
	return texture_create_from_pixels(u32(img.width), u32(img.height), img.pixels.buf[:], generate_mips), true
}

texture_create_from_file :: proc(path: string, generate_mips := true) -> (tex: GPU_Texture, ok: bool) {
	img, err := image.load_from_file(path, {.alpha_add_if_missing}, context.temp_allocator)
	if err != nil || img == nil do return {}, false
	defer image.destroy(img, context.temp_allocator)
	return texture_create_from_pixels(u32(img.width), u32(img.height), img.pixels.buf[:], generate_mips), true
}

mesh_create :: proc(verts: []Vertex, indices: []u32) -> (mesh: GPU_Mesh) {
	v_count := u32(len(verts))
	assert(vks.vertex_count_total + v_count <= MAX_VERTICES, "Vertex buffer overflow: MAX_VERTICES exceeded")
	mesh.vertex_offset = i32(vks.vertex_count_total)
	vks.vertex_count_total += v_count

	mesh.index_count = u32(len(indices))
	assert(vks.index_count_total + mesh.index_count <= MAX_INDICES, "Index buffer overflow: MAX_INDICES exceeded")
	mesh.index_offset = vks.index_count_total
	vks.index_count_total += mesh.index_count

	v_size := vk.DeviceSize(v_count * size_of(Vertex))
	i_size := vk.DeviceSize(mesh.index_count * size_of(u32))

	total_staging := v_size + i_size
	staging_ptr, staging_off := staging_alloc(total_staging, 16)
	v_off := staging_off
	i_off := staging_off + v_size

	mem.copy(staging_ptr, raw_data(verts), int(v_size))
	mem.copy(rawptr(uintptr(staging_ptr) + uintptr(v_size)), raw_data(indices), int(i_size))

	v_copy := vk.BufferCopy{
		srcOffset = v_off,
		dstOffset = vk.DeviceSize(mesh.vertex_offset * size_of(Vertex)),
		size      = v_size,
	}
	vk.CmdCopyBuffer(vks.upload_cmd, vks.staging_buffer.handle, vks.vertex_buffer.handle, 1, &v_copy)

	i_copy := vk.BufferCopy{
		srcOffset = i_off,
		dstOffset = vk.DeviceSize(mesh.index_offset * size_of(u32)),
		size      = i_size,
	}
	vk.CmdCopyBuffer(vks.upload_cmd, vks.staging_buffer.handle, vks.index_buffer.handle, 1, &i_copy)

	barriers := [?]vk.BufferMemoryBarrier2{
		{
			sType               = .BUFFER_MEMORY_BARRIER_2,
			srcStageMask        = {.TRANSFER},
			srcAccessMask       = {.TRANSFER_WRITE},
			dstStageMask        = {.ACCELERATION_STRUCTURE_BUILD_KHR, .VERTEX_ATTRIBUTE_INPUT},
			dstAccessMask       = {.ACCELERATION_STRUCTURE_READ_KHR, .VERTEX_ATTRIBUTE_READ},
			srcQueueFamilyIndex = vk.QUEUE_FAMILY_IGNORED,
			dstQueueFamilyIndex = vk.QUEUE_FAMILY_IGNORED,
			buffer              = vks.vertex_buffer.handle,
			offset              = vk.DeviceSize(mesh.vertex_offset * size_of(Vertex)),
			size                = v_size,
		},
		{
			sType               = .BUFFER_MEMORY_BARRIER_2,
			srcStageMask        = {.TRANSFER},
			srcAccessMask       = {.TRANSFER_WRITE},
			dstStageMask        = {.ACCELERATION_STRUCTURE_BUILD_KHR, .INDEX_INPUT},
			dstAccessMask       = {.ACCELERATION_STRUCTURE_READ_KHR, .INDEX_READ},
			srcQueueFamilyIndex = vk.QUEUE_FAMILY_IGNORED,
			dstQueueFamilyIndex = vk.QUEUE_FAMILY_IGNORED,
			buffer              = vks.index_buffer.handle,
			offset              = vk.DeviceSize(mesh.index_offset * size_of(u32)),
			size                = i_size,
		},
	}
	dep_info := vk.DependencyInfo{
		sType                    = .DEPENDENCY_INFO,
		bufferMemoryBarrierCount = len(barriers),
		pBufferMemoryBarriers    = raw_data(barriers[:]),
	}
	vk.CmdPipelineBarrier2(vks.upload_cmd, &dep_info)

	triangles := vk.AccelerationStructureGeometryTrianglesDataKHR{
		sType        = .ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR,
		vertexFormat = .R32G32B32_SFLOAT,
		vertexData   = { deviceAddress = vks.vertex_buffer.address + vk.DeviceAddress(mesh.vertex_offset * size_of(Vertex)) },
		vertexStride = size_of(Vertex),
		maxVertex    = v_count - 1,
		indexType    = .UINT32,
		indexData    = { deviceAddress = vks.index_buffer.address + vk.DeviceAddress(mesh.index_offset * size_of(u32)) },
	}
	geom := vk.AccelerationStructureGeometryKHR{
		sType        = .ACCELERATION_STRUCTURE_GEOMETRY_KHR,
		geometryType = .TRIANGLES,
		geometry     = { triangles = triangles },
		flags        = {.OPAQUE},
	}
	build_info := vk.AccelerationStructureBuildGeometryInfoKHR{
		sType         = .ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR,
		type          = .BOTTOM_LEVEL,
		flags         = {.PREFER_FAST_TRACE},
		mode          = .BUILD,
		geometryCount = 1,
		pGeometries   = &geom,
	}
	primitive_count := mesh.index_count / 3
	size_info := vk.AccelerationStructureBuildSizesInfoKHR{
		sType = .ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR,
	}
	vk.GetAccelerationStructureBuildSizesKHR(vks.device, .DEVICE, &build_info, &primitive_count, &size_info)

	blas_buf := buffer_create(size_info.accelerationStructureSize, {.ACCELERATION_STRUCTURE_STORAGE_KHR, .SHADER_DEVICE_ADDRESS})
	as_create_info := vk.AccelerationStructureCreateInfoKHR{
		sType  = .ACCELERATION_STRUCTURE_CREATE_INFO_KHR,
		buffer = blas_buf.handle,
		size   = size_info.accelerationStructureSize,
		type   = .BOTTOM_LEVEL,
	}
	blas: vk.AccelerationStructureKHR
	vk.CreateAccelerationStructureKHR(vks.device, &as_create_info, nil, &blas)

	build_info.dstAccelerationStructure = blas
	build_info.scratchData = { deviceAddress = vks.blas_scratch_buffer.address }

	range_info := vk.AccelerationStructureBuildRangeInfoKHR{
		primitiveCount = primitive_count,
	}

	scratch_barrier := vk.MemoryBarrier2{
		sType         = .MEMORY_BARRIER_2,
		srcStageMask  = {.ACCELERATION_STRUCTURE_BUILD_KHR},
		srcAccessMask = {.ACCELERATION_STRUCTURE_WRITE_KHR, .ACCELERATION_STRUCTURE_READ_KHR},
		dstStageMask  = {.ACCELERATION_STRUCTURE_BUILD_KHR},
		dstAccessMask = {.ACCELERATION_STRUCTURE_WRITE_KHR, .ACCELERATION_STRUCTURE_READ_KHR},
	}
	scratch_dep := vk.DependencyInfo{
		sType              = .DEPENDENCY_INFO,
		memoryBarrierCount = 1,
		pMemoryBarriers    = &scratch_barrier,
	}
	vk.CmdPipelineBarrier2(vks.upload_cmd, &scratch_dep)

	p_range := cast([^]vk.AccelerationStructureBuildRangeInfoKHR)&range_info
	vk.CmdBuildAccelerationStructuresKHR(vks.upload_cmd, 1, &build_info, &p_range)

	addr_info := vk.AccelerationStructureDeviceAddressInfoKHR{
		sType                 = .ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR,
		accelerationStructure = blas,
	}
	mesh.blas_address = vk.GetAccelerationStructureDeviceAddressKHR(vks.device, &addr_info)

	return mesh
}

mesh_draw :: proc(
	m:       GPU_Mesh,
	model:   Mat4        = linalg.MATRIX4F32_IDENTITY,
	color:   Color       = {255, 255, 255, 255},
	texture: GPU_Texture = {},
) {
	assert(len(vks.gpu_instances) < MAX_INSTANCES, "Instance buffer overflow: MAX_INSTANCES exceeded")
	draw_id := u32(len(vks.gpu_instances))

	append(&vks.gpu_instances, Gpu_Instance{
		model_mat  = model,
		color      = color,
		texture    = texture.index,
	})

	append(&vks.blas_addresses, m.blas_address)

	if len(vks.indirect_cmds) > 0 {
		last_cmd := &vks.indirect_cmds[len(vks.indirect_cmds) - 1]
		if last_cmd.firstIndex == m.index_offset &&
		   last_cmd.vertexOffset == m.vertex_offset &&
		   last_cmd.indexCount == m.index_count {
			last_cmd.instanceCount += 1
			return
		}
	}

	append(&vks.indirect_cmds, vk.DrawIndexedIndirectCommand{
		indexCount    = m.index_count,
		instanceCount = 1,
		firstIndex    = m.index_offset,
		vertexOffset  = m.vertex_offset,
		firstInstance = draw_id,
	})
}

camera_set :: proc(pos: Vec3, target: Vec3, up: Vec3 = {0, 1, 0}, fov := f32(90), near := f32(0.01), far := f32(10000)) {
	camera = {
		pos    = pos,
		target = target,
		up     = up,
		fov    = fov,
		near   = near,
		far    = far,
	}
}

skybox_set :: proc(tex: GPU_Texture) {
	vks.sky_texture = tex
}

pipeline_init :: proc() {
	VERT_SPV := #load("../assets/shaders/shader.vert.spv", []u32)
	FRAG_SPV := #load("../assets/shaders/shader.frag.spv", []u32)

	pool_sizes := [?]vk.DescriptorPoolSize{
		{ type = .COMBINED_IMAGE_SAMPLER, descriptorCount = MAX_TEXTURES },
		{ type = .ACCELERATION_STRUCTURE_KHR, descriptorCount = 1 },
	}
	desc_pool_info := vk.DescriptorPoolCreateInfo{
		sType         = .DESCRIPTOR_POOL_CREATE_INFO,
		flags         = {.UPDATE_AFTER_BIND},
		maxSets       = 1,
		poolSizeCount = u32(len(pool_sizes)),
		pPoolSizes    = raw_data(pool_sizes[:]),
	}
	desc_pool: vk.DescriptorPool
	vk.CreateDescriptorPool(vks.device, &desc_pool_info, nil, &desc_pool)

	binding_flags := [?]vk.DescriptorBindingFlags{
		{.PARTIALLY_BOUND, .UPDATE_AFTER_BIND},
		{},
	}
	flags_info := vk.DescriptorSetLayoutBindingFlagsCreateInfo{
		sType         = .DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO,
		bindingCount  = u32(len(binding_flags)),
		pBindingFlags = raw_data(binding_flags[:]),
	}
	bindings := [?]vk.DescriptorSetLayoutBinding{
		{
			binding         = 0,
			descriptorType  = .COMBINED_IMAGE_SAMPLER,
			descriptorCount = MAX_TEXTURES,
			stageFlags      = {.FRAGMENT},
		},
		{
			binding         = 1,
			descriptorType  = .ACCELERATION_STRUCTURE_KHR,
			descriptorCount = 1,
			stageFlags      = {.FRAGMENT},
		},
	}
	layout_info := vk.DescriptorSetLayoutCreateInfo{
		sType        = .DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
		pNext        = &flags_info,
		flags        = {.UPDATE_AFTER_BIND_POOL},
		bindingCount = u32(len(bindings)),
		pBindings    = raw_data(bindings[:]),
	}
	desc_layout: vk.DescriptorSetLayout
	vk.CreateDescriptorSetLayout(vks.device, &layout_info, nil, &desc_layout)

	alloc_set := vk.DescriptorSetAllocateInfo{
		sType              = .DESCRIPTOR_SET_ALLOCATE_INFO,
		descriptorPool     = desc_pool,
		descriptorSetCount = 1,
		pSetLayouts        = &desc_layout,
	}
	vk.AllocateDescriptorSets(vks.device, &alloc_set, &vks.descriptor_set)

	push_range := vk.PushConstantRange{
		stageFlags = {.VERTEX},
		size       = size_of(Push_Constants),
	}
	pipeline_layout_info := vk.PipelineLayoutCreateInfo{
		sType                  = .PIPELINE_LAYOUT_CREATE_INFO,
		setLayoutCount         = 1,
		pSetLayouts            = &desc_layout,
		pushConstantRangeCount = 1,
		pPushConstantRanges    = &push_range,
	}
	vk.CreatePipelineLayout(vks.device, &pipeline_layout_info, nil, &vks.pipeline_layout)

	vert_info := vk.ShaderModuleCreateInfo{
		sType    = .SHADER_MODULE_CREATE_INFO,
		codeSize = len(VERT_SPV) * size_of(u32),
		pCode    = raw_data(VERT_SPV),
	}
	vert_module: vk.ShaderModule
	if vk.CreateShaderModule(vks.device, &vert_info, nil, &vert_module) != .SUCCESS {
		panic("Failed to compile vertex shader module")
	}

	frag_info := vk.ShaderModuleCreateInfo{
		sType    = .SHADER_MODULE_CREATE_INFO,
		codeSize = len(FRAG_SPV) * size_of(u32),
		pCode    = raw_data(FRAG_SPV),
	}
	frag_module: vk.ShaderModule
	if vk.CreateShaderModule(vks.device, &frag_info, nil, &frag_module) != .SUCCESS {
		panic("Failed to compile fragment shader module")
	}

	stages := [?]vk.PipelineShaderStageCreateInfo{
		{ sType = .PIPELINE_SHADER_STAGE_CREATE_INFO, stage = {.VERTEX}, module = vert_module, pName = "main" },
		{ sType = .PIPELINE_SHADER_STAGE_CREATE_INFO, stage = {.FRAGMENT}, module = frag_module, pName = "main" },
	}

	binding_desc := vk.VertexInputBindingDescription{
		binding   = 0,
		stride    = size_of(Vertex),
		inputRate = .VERTEX,
	}
	attrib_descs := [4]vk.VertexInputAttributeDescription{
		{ location = 0, binding = 0, format = .R32G32B32_SFLOAT, offset = u32(offset_of(Vertex, pos)) },
		{ location = 1, binding = 0, format = .R32G32B32_SFLOAT, offset = u32(offset_of(Vertex, normal)) },
		{ location = 2, binding = 0, format = .R32G32_SFLOAT,    offset = u32(offset_of(Vertex, uv)) },
		{ location = 3, binding = 0, format = .R8G8B8A8_UNORM,   offset = u32(offset_of(Vertex, color)) },
	}
	vertex_input := vk.PipelineVertexInputStateCreateInfo{
		sType                           = .PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
		vertexBindingDescriptionCount   = 1,
		pVertexBindingDescriptions      = &binding_desc,
		vertexAttributeDescriptionCount = u32(len(attrib_descs)),
		pVertexAttributeDescriptions    = raw_data(attrib_descs[:]),
	}
	input_assembly := vk.PipelineInputAssemblyStateCreateInfo{
		sType    = .PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
		topology = .TRIANGLE_LIST,
	}
	viewport_state := vk.PipelineViewportStateCreateInfo{
		sType         = .PIPELINE_VIEWPORT_STATE_CREATE_INFO,
		viewportCount = 1,
		scissorCount  = 1,
	}
	rasterizer := vk.PipelineRasterizationStateCreateInfo{
		sType       = .PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
		polygonMode = .FILL,
		lineWidth   = 1.0,
		cullMode    = {.BACK},
		frontFace   = .COUNTER_CLOCKWISE,
	}
	multisampling := vk.PipelineMultisampleStateCreateInfo{
		sType                = .PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
		rasterizationSamples = {._4},
	}
	dynamic_states := [2]vk.DynamicState{ .VIEWPORT, .SCISSOR }
	dynamic_state := vk.PipelineDynamicStateCreateInfo{
		sType             = .PIPELINE_DYNAMIC_STATE_CREATE_INFO,
		dynamicStateCount = u32(len(dynamic_states)),
		pDynamicStates    = raw_data(dynamic_states[:]),
	}
	depth_stencil := vk.PipelineDepthStencilStateCreateInfo{
		sType            = .PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
		depthTestEnable  = true,
		depthWriteEnable = true,
		depthCompareOp   = .LESS_OR_EQUAL,
	}
	color_blend_attachment := vk.PipelineColorBlendAttachmentState{
		blendEnable         = true,
		srcColorBlendFactor = .SRC_ALPHA,
		dstColorBlendFactor = .ONE_MINUS_SRC_ALPHA,
		colorBlendOp        = .ADD,
		srcAlphaBlendFactor = .ONE,
		dstAlphaBlendFactor = .ONE_MINUS_SRC_ALPHA,
		alphaBlendOp        = .ADD,
		colorWriteMask      = {.R, .G, .B, .A},
	}
	color_blending := vk.PipelineColorBlendStateCreateInfo{
		sType           = .PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
		attachmentCount = 1,
		pAttachments    = &color_blend_attachment,
	}

	swapchain_fmt := vk.Format.B8G8R8A8_SRGB
	pipeline_rendering_info := vk.PipelineRenderingCreateInfo{
		sType                   = .PIPELINE_RENDERING_CREATE_INFO,
		colorAttachmentCount    = 1,
		pColorAttachmentFormats = &swapchain_fmt,
		depthAttachmentFormat   = .D32_SFLOAT,
	}
	pipeline_info := vk.GraphicsPipelineCreateInfo{
		sType               = .GRAPHICS_PIPELINE_CREATE_INFO,
		pNext               = &pipeline_rendering_info,
		stageCount          = len(stages),
		pStages             = raw_data(stages[:]),
		pVertexInputState   = &vertex_input,
		pInputAssemblyState = &input_assembly,
		pViewportState      = &viewport_state,
		pRasterizationState = &rasterizer,
		pMultisampleState   = &multisampling,
		pDepthStencilState  = &depth_stencil,
		pColorBlendState    = &color_blending,
		pDynamicState       = &dynamic_state,
		layout              = vks.pipeline_layout,
	}

	if vk.CreateGraphicsPipelines(vks.device, 0, 1, &pipeline_info, nil, &vks.pipeline) != .SUCCESS {
		panic("Failed to create graphics pipeline")
	}

	vk.DestroyShaderModule(vks.device, vert_module, nil)
	vk.DestroyShaderModule(vks.device, frag_module, nil)
}

vk_swapchain_create :: proc() {
	vk.DeviceWaitIdle(vks.device)

	caps: vk.SurfaceCapabilitiesKHR
	vk.GetPhysicalDeviceSurfaceCapabilitiesKHR(vks.gpu, vks.surface, &caps)

	extent := vk.Extent2D{
		width  = clamp(u32(window.size.x), caps.minImageExtent.width, caps.maxImageExtent.width),
		height = clamp(u32(window.size.y), caps.minImageExtent.height, caps.maxImageExtent.height),
	}

	image_count := caps.minImageCount + 1
	if caps.maxImageCount > 0 && image_count > caps.maxImageCount {
		image_count = caps.maxImageCount
	}

	old_swapchain := vks.swapchain
	swapchain_info := vk.SwapchainCreateInfoKHR{
		sType            = .SWAPCHAIN_CREATE_INFO_KHR,
		surface          = vks.surface,
		minImageCount    = image_count,
		imageFormat      = .B8G8R8A8_SRGB,
		imageColorSpace  = .SRGB_NONLINEAR,
		imageExtent      = extent,
		imageArrayLayers = 1,
		imageUsage       = {.COLOR_ATTACHMENT, .TRANSFER_DST},
		imageSharingMode = .EXCLUSIVE,
		preTransform     = caps.currentTransform,
		compositeAlpha   = {.OPAQUE},
		presentMode      = .FIFO,
		clipped          = true,
		oldSwapchain     = old_swapchain,
	}
	vk.CreateSwapchainKHR(vks.device, &swapchain_info, nil, &vks.swapchain)

	if old_swapchain != 0 {
		for view in vks.swapchain_views {
			vk.DestroyImageView(vks.device, view, nil)
		}
		delete(vks.swapchain_views)
		delete(vks.swapchain_images)
		vk.DestroySwapchainKHR(vks.device, old_swapchain, nil)
	}

	texture_destroy(&vks.depth_image)
	texture_destroy(&vks.msaa_image)

	actual_count: u32
	vk.GetSwapchainImagesKHR(vks.device, vks.swapchain, &actual_count, nil)
	vks.swapchain_images = make([]vk.Image, actual_count)
	vk.GetSwapchainImagesKHR(vks.device, vks.swapchain, &actual_count, raw_data(vks.swapchain_images))

	vks.swapchain_views = make([]vk.ImageView, actual_count)
	for i in 0..<actual_count {
		view_info := vk.ImageViewCreateInfo{
			sType    = .IMAGE_VIEW_CREATE_INFO,
			image    = vks.swapchain_images[i],
			viewType = .D2,
			format   = .B8G8R8A8_SRGB,
			subresourceRange = {
				aspectMask = {.COLOR},
				levelCount = 1,
				layerCount = 1,
			},
		}
		vk.CreateImageView(vks.device, &view_info, nil, &vks.swapchain_views[i])
	}

	for sem in vks.render_done {
		vk.DestroySemaphore(vks.device, sem, nil)
	}
	delete(vks.render_done)
	vks.render_done = make([]vk.Semaphore, actual_count)
	for i in 0..<actual_count {
		sem_info := vk.SemaphoreCreateInfo{ sType = .SEMAPHORE_CREATE_INFO }
		vk.CreateSemaphore(vks.device, &sem_info, nil, &vks.render_done[i])
	}

	vks.msaa_image = texture_create(
		extent.width, extent.height,
		.B8G8R8A8_SRGB,
		{.COLOR_ATTACHMENT, .TRANSIENT_ATTACHMENT},
		{._4}, 1, {.COLOR},
	)

	vks.depth_image = texture_create(
		extent.width, extent.height,
		.D32_SFLOAT,
		{.DEPTH_STENCIL_ATTACHMENT, .TRANSIENT_ATTACHMENT},
		{._4}, 1, {.DEPTH},
	)
}

vk_init :: proc() {
	lib_name := "vulkan-1.dll" when ODIN_OS == .Windows else "libvulkan.so.1"
	lib := dynlib.load_library(lib_name) or_else panic("Failed to load Vulkan library")
	vk_get_instance_proc_addr := dynlib.symbol_address(lib, "vkGetInstanceProcAddr")
	vk.load_proc_addresses_global(vk_get_instance_proc_addr)

	app_info := vk.ApplicationInfo{
		sType      = .APPLICATION_INFO,
		apiVersion = vk.API_VERSION_1_3,
	}
	instance_extensions := vk_platform_instance_extensions()
	instance_info := vk.InstanceCreateInfo{
		sType                   = .INSTANCE_CREATE_INFO,
		pApplicationInfo        = &app_info,
		enabledExtensionCount   = u32(len(instance_extensions)),
		ppEnabledExtensionNames = raw_data(instance_extensions),
	}
	instance: vk.Instance
	vk.CreateInstance(&instance_info, nil, &instance)
	vk.load_proc_addresses_instance(instance)

	vks.surface = vk_create_platform_surface(instance) or_else panic("Failed to create surface")

	device_count: u32
	vk.EnumeratePhysicalDevices(instance, &device_count, nil)
	devices := make([]vk.PhysicalDevice, device_count, context.temp_allocator)
	vk.EnumeratePhysicalDevices(instance, &device_count, raw_data(devices))

	vks.gpu = devices[0]
	for d in devices {
		props: vk.PhysicalDeviceProperties
		vk.GetPhysicalDeviceProperties(d, &props)
		if props.deviceType == .DISCRETE_GPU {
			vks.gpu = d
			break
		}
	}

	queue_count: u32
	vk.GetPhysicalDeviceQueueFamilyProperties(vks.gpu, &queue_count, nil)
	queue_props := make([]vk.QueueFamilyProperties, queue_count, context.temp_allocator)
	vk.GetPhysicalDeviceQueueFamilyProperties(vks.gpu, &queue_count, &queue_props[0])

	for qp, idx in queue_props {
		present_support: b32
		vk.GetPhysicalDeviceSurfaceSupportKHR(vks.gpu, u32(idx), vks.surface, &present_support)
		if .GRAPHICS in qp.queueFlags && present_support {
			vks.queue_family = u32(idx)
			break
		}
	}

	queue_priority := f32(1.0)
	queue_create_info := vk.DeviceQueueCreateInfo{
		sType            = .DEVICE_QUEUE_CREATE_INFO,
		queueFamilyIndex = vks.queue_family,
		queueCount       = 1,
		pQueuePriorities = &queue_priority,
	}

	accel_features := vk.PhysicalDeviceAccelerationStructureFeaturesKHR{
		sType                 = .PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR,
		accelerationStructure = true,
	}
	ray_query_features := vk.PhysicalDeviceRayQueryFeaturesKHR{
		sType    = .PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR,
		pNext    = &accel_features,
		rayQuery = true,
	}
	features13 := vk.PhysicalDeviceVulkan13Features{
		sType            = .PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
		pNext            = &ray_query_features,
		dynamicRendering = true,
		synchronization2 = true,
	}
	features12 := vk.PhysicalDeviceVulkan12Features{
		sType                                        = .PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
		pNext                                        = &features13,
		bufferDeviceAddress                          = true,
		descriptorIndexing                           = true,
		runtimeDescriptorArray                       = true,
		shaderSampledImageArrayNonUniformIndexing    = true,
		descriptorBindingPartiallyBound              = true,
		descriptorBindingSampledImageUpdateAfterBind = true,
		scalarBlockLayout                            = true,
	}
	features2 := vk.PhysicalDeviceFeatures2{
		sType    = .PHYSICAL_DEVICE_FEATURES_2,
		pNext    = &features12,
		features = {
			samplerAnisotropy = true,
			multiDrawIndirect = true,
		},
	}

	device_extensions := [?]cstring{
		vk.KHR_SWAPCHAIN_EXTENSION_NAME,
		vk.KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,
		vk.KHR_RAY_QUERY_EXTENSION_NAME,
		vk.KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME,
	}
	device_info := vk.DeviceCreateInfo{
		sType                   = .DEVICE_CREATE_INFO,
		pNext                   = &features2,
		queueCreateInfoCount    = 1,
		pQueueCreateInfos       = &queue_create_info,
		enabledExtensionCount   = u32(len(device_extensions)),
		ppEnabledExtensionNames = raw_data(device_extensions[:]),
	}
	vk.CreateDevice(vks.gpu, &device_info, nil, &vks.device)
	vk.load_proc_addresses_device(vks.device)
	vk.GetDeviceQueue(vks.device, vks.queue_family, 0, &vks.queue)

	cmd_pool_info := vk.CommandPoolCreateInfo{
		sType            = .COMMAND_POOL_CREATE_INFO,
		flags            = {.RESET_COMMAND_BUFFER},
		queueFamilyIndex = vks.queue_family,
	}
	vk.CreateCommandPool(vks.device, &cmd_pool_info, nil, &vks.cmd_pool)
	vk.CreateCommandPool(vks.device, &cmd_pool_info, nil, &vks.upload_cmd_pool)

	alloc_cmd := vk.CommandBufferAllocateInfo{
		sType              = .COMMAND_BUFFER_ALLOCATE_INFO,
		commandPool        = vks.cmd_pool,
		level              = .PRIMARY,
		commandBufferCount = 1,
	}
	vk.AllocateCommandBuffers(vks.device, &alloc_cmd, &vks.cmd_buf)

	sem_info := vk.SemaphoreCreateInfo{ sType = .SEMAPHORE_CREATE_INFO }
	fence_info := vk.FenceCreateInfo{ sType = .FENCE_CREATE_INFO, flags = {.SIGNALED} }
	vk.CreateSemaphore(vks.device, &sem_info, nil, &vks.image_acquired)
	vk.CreateFence(vks.device, &fence_info, nil, &vks.in_flight)

	sampler_info := vk.SamplerCreateInfo{
		sType            = .SAMPLER_CREATE_INFO,
		magFilter        = .LINEAR,
		minFilter        = .LINEAR,
		mipmapMode       = .LINEAR,
		addressModeU     = .REPEAT,
		addressModeV     = .REPEAT,
		addressModeW     = .REPEAT,
		anisotropyEnable = true,
		maxAnisotropy    = 16.0,
		minLod           = 0.0,
		maxLod           = vk.LOD_CLAMP_NONE,
		borderColor      = .INT_OPAQUE_BLACK,
	}
	vk.CreateSampler(vks.device, &sampler_info, nil, &vks.sampler)

	pipeline_init()
	vk_swapchain_create()

	vks.vertex_buffer = buffer_create(vk.DeviceSize(MAX_VERTICES * size_of(Vertex)), {.VERTEX_BUFFER, .ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR, .TRANSFER_DST})
	vks.index_buffer  = buffer_create(vk.DeviceSize(MAX_INDICES * size_of(u32)), {.INDEX_BUFFER, .ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR, .TRANSFER_DST})
	vks.instance_buffer = buffer_create(vk.DeviceSize(MAX_INSTANCES * size_of(Gpu_Instance)), {.STORAGE_BUFFER}, host_visible = true)
	vks.indirect_buffer = buffer_create(vk.DeviceSize(MAX_INSTANCES * size_of(vk.DrawIndexedIndirectCommand)), {.INDIRECT_BUFFER}, host_visible = true)
	vks.staging_buffer  = buffer_create(STAGING_SIZE, {.TRANSFER_SRC}, host_visible = true)
	vks.blas_scratch_buffer = buffer_create(BLAS_SIZE, {.STORAGE_BUFFER, .SHADER_DEVICE_ADDRESS})
	vks.tlas_instances_buffer = buffer_create(vk.DeviceSize(MAX_INSTANCES * size_of(vk.AccelerationStructureInstanceKHR)), {.ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR, .SHADER_DEVICE_ADDRESS}, host_visible = true)

	tlas_geom_test := vk.AccelerationStructureGeometryKHR{
		sType        = .ACCELERATION_STRUCTURE_GEOMETRY_KHR,
		geometryType = .INSTANCES,
		geometry     = { instances = { sType = .ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR } },
	}
	tlas_build_info_size := vk.AccelerationStructureBuildGeometryInfoKHR{
		sType         = .ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR,
		type          = .TOP_LEVEL,
		flags         = {.PREFER_FAST_BUILD},
		mode          = .BUILD,
		geometryCount = 1,
		pGeometries   = &tlas_geom_test,
	}

	max_inst := u32(MAX_INSTANCES)
	tlas_sizes := vk.AccelerationStructureBuildSizesInfoKHR{ sType = .ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR }
	vk.GetAccelerationStructureBuildSizesKHR(vks.device, .DEVICE, &tlas_build_info_size, &max_inst, &tlas_sizes)

	vks.tlas_buffer = buffer_create(tlas_sizes.accelerationStructureSize, {.ACCELERATION_STRUCTURE_STORAGE_KHR, .SHADER_DEVICE_ADDRESS})
	vks.tlas_scratch_buffer = buffer_create(tlas_sizes.buildScratchSize, {.STORAGE_BUFFER, .SHADER_DEVICE_ADDRESS})

	tlas_create_info := vk.AccelerationStructureCreateInfoKHR{
		sType  = .ACCELERATION_STRUCTURE_CREATE_INFO_KHR,
		buffer = vks.tlas_buffer.handle,
		size   = tlas_sizes.accelerationStructureSize,
		type   = .TOP_LEVEL,
	}
	vk.CreateAccelerationStructureKHR(vks.device, &tlas_create_info, nil, &vks.tlas)

	tlas_desc_info := vk.WriteDescriptorSetAccelerationStructureKHR{
		sType                      = .WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR,
		accelerationStructureCount = 1,
		pAccelerationStructures    = &vks.tlas,
	}
	write_tlas := vk.WriteDescriptorSet{
		sType           = .WRITE_DESCRIPTOR_SET,
		pNext           = &tlas_desc_info,
		dstSet          = vks.descriptor_set,
		dstBinding      = 1,
		dstArrayElement = 0,
		descriptorType  = .ACCELERATION_STRUCTURE_KHR,
		descriptorCount = 1,
	}
	vk.UpdateDescriptorSets(vks.device, 1, &write_tlas, 0, nil)

	upload_begin()
	texture_create_from_pixels(1, 1, {255, 255, 255, 255}, false)

	sky_verts := [3]Vertex{
		{ pos = {-1, -1, 1}, uv = {0, 0}, color = {255, 255, 255, 255} },
		{ pos = { 3, -1, 1}, uv = {2, 0}, color = {255, 255, 255, 255} },
		{ pos = {-1,  3, 1}, uv = {0, 2}, color = {255, 255, 255, 255} },
	}
	sky_indices := [3]u32{0, 1, 2}
	vks.sky_mesh = mesh_create(sky_verts[:], sky_indices[:])
	upload_end()
	camera_set({1, 1, 1}, {0, 0, 0})
}

vk_render :: proc() {
	vk.WaitForFences(vks.device, 1, &vks.in_flight, true, max(u64))

	image_idx: u32
	res := vk.AcquireNextImageKHR(vks.device, vks.swapchain, max(u64), vks.image_acquired, 0, &image_idx)
	if res == .ERROR_OUT_OF_DATE_KHR {
		vk_swapchain_create()
		return
	} else if res != .SUCCESS && res != .SUBOPTIMAL_KHR {
		return
	}

	vk.ResetFences(vks.device, 1, &vks.in_flight)

	aspect := f32(window.size.x) / f32(max(window.size.y, 1))
	proj := linalg.matrix4_perspective_f32(math.to_radians_f32(camera.fov), aspect, camera.near, camera.far)
	view := linalg.matrix4_look_at_f32(camera.pos, camera.target, camera.up)

	if vks.sky_texture.view != 0 {
		view_rot := linalg.matrix4_look_at_f32({0, 0, 0}, camera.target - camera.pos, camera.up)
		inv_view_proj := linalg.matrix4_inverse(proj * view_rot)

		sky_id := u32(len(vks.gpu_instances))
		append(&vks.gpu_instances, Gpu_Instance{
			model_mat = inv_view_proj,
			color     = {255, 255, 255, 255},
			texture   = vks.sky_texture.index,
			type      = 1,
		})
		append(&vks.indirect_cmds, vk.DrawIndexedIndirectCommand{
			indexCount    = vks.sky_mesh.index_count,
			instanceCount = 1,
			firstIndex    = vks.sky_mesh.index_offset,
			vertexOffset  = vks.sky_mesh.vertex_offset,
			firstInstance = sky_id,
		})
	}

	mem.copy(vks.instance_buffer.mapped, raw_data(vks.gpu_instances[:]), len(vks.gpu_instances) * size_of(Gpu_Instance))
	mem.copy(vks.indirect_buffer.mapped, raw_data(vks.indirect_cmds[:]), len(vks.indirect_cmds) * size_of(vk.DrawIndexedIndirectCommand))

	vk.ResetCommandBuffer(vks.cmd_buf, {})
	begin_info := vk.CommandBufferBeginInfo{
		sType = .COMMAND_BUFFER_BEGIN_INFO,
		flags = {.ONE_TIME_SUBMIT},
	}
	vk.BeginCommandBuffer(vks.cmd_buf, &begin_info)

	as_count := u32(len(vks.blas_addresses))
	if as_count > 0 {
		as_instances := (cast([^]vk.AccelerationStructureInstanceKHR)vks.tlas_instances_buffer.mapped)[:as_count]
		for &inst, i in as_instances {
			m := vks.gpu_instances[i].model_mat
			inst = vk.AccelerationStructureInstanceKHR{
				transform = {
					mat = {
						{ m[0, 0], m[0, 1], m[0, 2], m[0, 3] },
						{ m[1, 0], m[1, 1], m[1, 2], m[1, 3] },
						{ m[2, 0], m[2, 1], m[2, 2], m[2, 3] },
					},
				},
				instanceCustomIndex            = u32(i),
				mask                           = 0xFF,
				flags                          = .TRIANGLE_FACING_CULL_DISABLE,
				accelerationStructureReference = u64(vks.blas_addresses[i]),
			}
		}

		tlas_geom := vk.AccelerationStructureGeometryKHR{
			sType        = .ACCELERATION_STRUCTURE_GEOMETRY_KHR,
			geometryType = .INSTANCES,
			geometry     = {
				instances = {
					sType = .ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR,
					data  = { deviceAddress = vks.tlas_instances_buffer.address },
				},
			},
		}
		tlas_build_info := vk.AccelerationStructureBuildGeometryInfoKHR{
			sType                    = .ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR,
			type                     = .TOP_LEVEL,
			flags                    = {.PREFER_FAST_BUILD},
			mode                     = .BUILD,
			dstAccelerationStructure = vks.tlas,
			geometryCount            = 1,
			pGeometries              = &tlas_geom,
			scratchData              = { deviceAddress = vks.tlas_scratch_buffer.address },
		}
		tlas_range := vk.AccelerationStructureBuildRangeInfoKHR{ primitiveCount = as_count }
		p_tlas_range := cast([^]vk.AccelerationStructureBuildRangeInfoKHR)&tlas_range
		vk.CmdBuildAccelerationStructuresKHR(vks.cmd_buf, 1, &tlas_build_info, &p_tlas_range)

		tlas_barrier := vk.MemoryBarrier2{
			sType         = .MEMORY_BARRIER_2,
			srcStageMask  = {.ACCELERATION_STRUCTURE_BUILD_KHR},
			srcAccessMask = {.ACCELERATION_STRUCTURE_WRITE_KHR},
			dstStageMask  = {.FRAGMENT_SHADER},
			dstAccessMask = {.ACCELERATION_STRUCTURE_READ_KHR},
		}
		dep_info := vk.DependencyInfo{
			sType              = .DEPENDENCY_INFO,
			memoryBarrierCount = 1,
			pMemoryBarriers    = &tlas_barrier,
		}
		vk.CmdPipelineBarrier2(vks.cmd_buf, &dep_info)
	}

	image_barrier(
		vks.cmd_buf, vks.msaa_image.image,
		.UNDEFINED, .COLOR_ATTACHMENT_OPTIMAL,
		{.COLOR_ATTACHMENT_OUTPUT}, {.COLOR_ATTACHMENT_OUTPUT},
		{}, {.COLOR_ATTACHMENT_WRITE},
		{.COLOR}, 1, 0,
	)
	image_barrier(
		vks.cmd_buf, vks.swapchain_images[image_idx],
		.UNDEFINED, .COLOR_ATTACHMENT_OPTIMAL,
		{.COLOR_ATTACHMENT_OUTPUT}, {.COLOR_ATTACHMENT_OUTPUT},
		{}, {.COLOR_ATTACHMENT_WRITE},
		{.COLOR}, 1, 0,
	)
	image_barrier(
		vks.cmd_buf, vks.depth_image.image,
		.UNDEFINED, .DEPTH_ATTACHMENT_OPTIMAL,
		{.EARLY_FRAGMENT_TESTS, .LATE_FRAGMENT_TESTS}, {.EARLY_FRAGMENT_TESTS, .LATE_FRAGMENT_TESTS},
		{}, {.DEPTH_STENCIL_ATTACHMENT_WRITE},
		{.DEPTH}, 1, 0,
	)

	extent := vk.Extent2D{ u32(window.size.x), u32(window.size.y) }
	viewport := vk.Viewport{
		x        = 0.0,
		y        = f32(extent.height),
		width    = f32(extent.width),
		height   = -f32(extent.height),
		minDepth = 0.0,
		maxDepth = 1.0,
	}
	scissor := vk.Rect2D{ extent = extent }

	color_attachment := vk.RenderingAttachmentInfo{
		sType              = .RENDERING_ATTACHMENT_INFO,
		imageView          = vks.msaa_image.view,
		imageLayout        = .COLOR_ATTACHMENT_OPTIMAL,
		resolveMode        = {.AVERAGE},
		resolveImageView   = vks.swapchain_views[image_idx],
		resolveImageLayout = .COLOR_ATTACHMENT_OPTIMAL,
		loadOp             = .CLEAR,
		storeOp            = .DONT_CARE,
		clearValue         = { color = { float32 = { 0.0, 0.0, 0.0, 1.0 } } },
	}
	depth_attachment := vk.RenderingAttachmentInfo{
		sType       = .RENDERING_ATTACHMENT_INFO,
		imageView   = vks.depth_image.view,
		imageLayout = .DEPTH_ATTACHMENT_OPTIMAL,
		loadOp      = .CLEAR,
		storeOp     = .DONT_CARE,
		clearValue  = { depthStencil = { depth = 1.0 } },
	}
	rendering_info := vk.RenderingInfo{
		sType                = .RENDERING_INFO,
		renderArea           = { extent = extent },
		layerCount           = 1,
		colorAttachmentCount = 1,
		pColorAttachments    = &color_attachment,
		pDepthAttachment     = &depth_attachment,
	}

	vk.CmdBeginRendering(vks.cmd_buf, &rendering_info)
	vk.CmdSetViewport(vks.cmd_buf, 0, 1, &viewport)
	vk.CmdSetScissor(vks.cmd_buf, 0, 1, &scissor)

	pc := Push_Constants{
		view_proj         = proj * view,
		instances_address = vks.instance_buffer.address,
	}
	vb_offset := vk.DeviceSize(0)

	vk.CmdBindPipeline(vks.cmd_buf, .GRAPHICS, vks.pipeline)
	vk.CmdBindDescriptorSets(vks.cmd_buf, .GRAPHICS, vks.pipeline_layout, 0, 1, &vks.descriptor_set, 0, nil)
	vk.CmdBindVertexBuffers(vks.cmd_buf, 0, 1, &vks.vertex_buffer.handle, &vb_offset)
	vk.CmdBindIndexBuffer(vks.cmd_buf, vks.index_buffer.handle, 0, .UINT32)
	vk.CmdPushConstants(vks.cmd_buf, vks.pipeline_layout, {.VERTEX}, 0, size_of(pc), &pc)

	vk.CmdDrawIndexedIndirect(
		vks.cmd_buf,
		vks.indirect_buffer.handle,
		0,
		u32(len(vks.indirect_cmds)),
		size_of(vk.DrawIndexedIndirectCommand),
	)

	vk.CmdEndRendering(vks.cmd_buf)

	image_barrier(
		vks.cmd_buf, vks.swapchain_images[image_idx],
		.COLOR_ATTACHMENT_OPTIMAL, .PRESENT_SRC_KHR,
		{.COLOR_ATTACHMENT_OUTPUT}, {.BOTTOM_OF_PIPE},
		{.COLOR_ATTACHMENT_WRITE}, {},
		{.COLOR}, 1, 0,
	)

	vk.EndCommandBuffer(vks.cmd_buf)

	cmd_submit := vk.CommandBufferSubmitInfo{
		sType         = .COMMAND_BUFFER_SUBMIT_INFO,
		commandBuffer = vks.cmd_buf,
	}
	wait_sem := vk.SemaphoreSubmitInfo{
		sType     = .SEMAPHORE_SUBMIT_INFO,
		semaphore = vks.image_acquired,
		stageMask = {.COLOR_ATTACHMENT_OUTPUT},
	}
	signal_sem := vk.SemaphoreSubmitInfo{
		sType     = .SEMAPHORE_SUBMIT_INFO,
		semaphore = vks.render_done[image_idx],
		stageMask = {.COLOR_ATTACHMENT_OUTPUT},
	}
	submit_info := vk.SubmitInfo2{
		sType                    = .SUBMIT_INFO_2,
		waitSemaphoreInfoCount   = 1,
		pWaitSemaphoreInfos      = &wait_sem,
		commandBufferInfoCount   = 1,
		pCommandBufferInfos      = &cmd_submit,
		signalSemaphoreInfoCount = 1,
		pSignalSemaphoreInfos    = &signal_sem,
	}
	vk.QueueSubmit2(vks.queue, 1, &submit_info, vks.in_flight)

	present_info := vk.PresentInfoKHR{
		sType              = .PRESENT_INFO_KHR,
		waitSemaphoreCount = 1,
		pWaitSemaphores    = &vks.render_done[image_idx],
		swapchainCount     = 1,
		pSwapchains        = &vks.swapchain,
		pImageIndices      = &image_idx,
	}
	pres_res := vk.QueuePresentKHR(vks.queue, &present_info)
	if pres_res == .ERROR_OUT_OF_DATE_KHR || pres_res == .SUBOPTIMAL_KHR || res == .SUBOPTIMAL_KHR {
		vk_swapchain_create()
	}

	clear(&vks.gpu_instances)
	clear(&vks.indirect_cmds)
	clear(&vks.blas_addresses)
}